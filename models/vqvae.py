import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.nn.functional import scaled_dot_product_attention

from models.quant import VectorQuantizerM
from models.vitamin import ViTaminDecoder, GeGluMlp


class PlainAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            # assert in_dim // num_heads == out_dim
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer('zero_k_bias', torch.zeros(in_dim))
        else:
            # assert out_dim // num_heads == in_dim
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer('zero_k_bias', torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        x = scaled_dot_product_attention(q, k, v)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = PlainAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(
            in_features=out_dim,
            hidden_features=hidden_dim
        )

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        # 1. build encoder
        self.encoder = timm.create_model(
            args.model,
            patch_size=1,
            fc_norm=True,
            drop_rate=0.0,
            num_classes=0,
            global_pool='',
            pos_embed='none',
            class_token=False,
            mlp_layer=GeGluMlp,
            img_size=args.img_size,
            drop_path_rate=args.drop_path,
        )
        self.encoder.set_grad_checkpointing(args.grad_ckpt)

        # 2. build conv before quant
        if args.quant_proj == 'linear':
            self.quant_proj = nn.Linear(self.encoder.embed_dim, args.vocab_width)
        elif args.quant_proj == 'attn':
            self.quant_proj = AttnProjection(self.encoder.embed_dim, args.vocab_width, args.num_codebooks)
        else:
            raise NotImplementedError

        # 3. build quant
        self.quantize = VectorQuantizerM(
            vocab_size=args.vocab_size,
            vocab_width=args.vocab_width,
            beta=args.vq_beta,
            use_entropy_loss=args.le > 0,
            entropy_temp=args.e_temp,
            num_codebooks=args.num_codebooks,
        )

        # 4. build conv after quant
        if args.quant_proj == 'linear':
            self.post_quant_proj = nn.Linear(args.vocab_width, self.encoder.embed_dim)
        elif args.quant_proj == 'attn':
            self.post_quant_proj = AttnProjection(args.vocab_width, self.encoder.embed_dim, args.num_codebooks)
        else:
            raise NotImplementedError

        # 5. build decoder
        self.decoder = ViTaminDecoder(
            args.model,
            depths=(4, 2),
            img_size=args.img_size,
            drop_path=args.drop_path,
            grad_ckpt=args.grad_ckpt
        )

        self.maybe_record_function = nullcontext

    def forward(self, img):
        features = self.encoder(img).float()
        with torch.cuda.amp.autocast(enabled=False):
            features = self.quant_proj(features)
            quant_out = self.quantize(features)
            features, vq_loss, entropy_loss, usages = quant_out
            features = self.post_quant_proj(features)
        rec_img = self.decoder(features).float()
        return rec_img, vq_loss, entropy_loss, usages

    def img_to_idx(self, img):
        features = self.encoder(img).float()
        features = self.quant_proj(features)
        return self.quantize.f_to_idx(features)

    def idx_to_img(self, indices):
        features = self.quantize.idx_to_f(indices)
        features = self.post_quant_proj(features)
        img = self.decoder(features).clamp_(-1, 1)
        return img

    def img_to_reconstructed_img(self, img) -> torch.Tensor:
        features = self.encoder(img).float()
        with torch.cuda.amp.autocast(enabled=False):
            features = self.quant_proj(features)
            quant_out = self.quantize(features)
            features, _, _, _ = quant_out
            features = self.post_quant_proj(features)
        rec_img = self.decoder(features).float().clamp_(-1, 1)
        return rec_img


if __name__ == '__main__':
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,
                nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)

    cnn = VQVAE(channel_num=64, vocab_norm=False)
    from models import init_weights

    init_weights(cnn, -0.5)
    torch.save(cnn.state_dict(), r'C:\Users\16333\Desktop\PyCharm\vlip\local_output\cnn.pth')
