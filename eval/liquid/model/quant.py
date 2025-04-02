from typing import List, Tuple
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention

# from utils import dist

# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer', ]

def get_entropy_loss(latent_embed, codebook_embed, inv_entropy_tau):
    E_dist = latent_embed.square().sum(dim=1, keepdim=True) + codebook_embed.square().sum(dim=1, keepdim=False)
    E_dist.addmm_(latent_embed, codebook_embed.T, alpha=-2, beta=1)  # E_dist: (N, vocab_size)
    logits = -E_dist.float().mul_(inv_entropy_tau)
    # calc per_sample_entropy
    prob, log_prob = logits.softmax(dim=-1), logits.log_softmax(dim=-1)  # both are (N, vocab_size)
    per_sample_entropy = torch.mean((-prob * log_prob).sum(dim=-1))
    # calc codebook_entropy
    avg_prob = prob.mean(dim=0)  # (vocab_size,)
    log_avg_prob = torch.log(avg_prob + 1e-7)
    codebook_entropy = (-avg_prob * log_avg_prob).sum()
    # calc entropy_loss
    entropy_loss = per_sample_entropy - codebook_entropy
    return entropy_loss


class NormalizedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # self.norm_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, idx):
        return F.embedding(
            idx, F.normalize(self.weight, dim=1), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )

    def get_norm_weight(self):
        return F.normalize(self.weight, dim=1)


class ResConv(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3 if quant_resi < 0 else 1
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class VectorQuantizer(nn.Module):
    def __init__(
            self, vocab_size: int, vocab_width: int, vocab_norm: bool, beta: float = 0.25, quant_resi=-0.5,
            using_entropy_loss=False, entropy_temp=0.01,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.vocab_width: int = vocab_width
        self.register_buffer('vocab_usage', torch.zeros(self.vocab_size))
        self.vocab_usage_record_times: int = 0

        self.vocab_norm: bool = vocab_norm
        # self.quant_resi = ResConv(self.vocab_width, quant_resi=quant_resi)
        self.quant_resi = nn.Identity()
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_width)
        self.beta: float = beta

        self.using_entropy_loss, self.inv_entropy_tau = using_entropy_loss, 1 / entropy_temp
        if not self.vocab_norm:
            assert not self.using_entropy_loss, 'entropy loss without vocab norm is not supported'

    def init_vocab(self, eini: float):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            base = self.vocab_width ** -0.5
            base /= 36
            self.embedding.weight.data.uniform_(-abs(eini) * base, abs(eini) * base)

    def extra_repr(self) -> str:
        return f'beta={self.beta:g}'

    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        f_BChw = f_BChw.float()
        B, C, h, w = f_BChw.shape
        if self.vocab_norm:
            if self.using_entropy_loss:
                # find the nearest neighbor
                NxC = f_BChw.permute(0, 2, 3, 1).reshape(-1, C)
                NxC_no_grad = NxC.detach()
                NxC_no_grad = F.normalize(NxC_no_grad, dim=-1)
                idx_N = torch.argmax(NxC_no_grad @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                # get logits
                E_dist = NxC.square().sum(dim=1, keepdim=True) + self.embedding.weight.square().sum(dim=1,
                                                                                                    keepdim=False)
                E_dist.addmm_(NxC, self.embedding.weight.T, alpha=-2, beta=1)  # E_dist: (N, vocab_size)
                logits = -E_dist.float().mul_(self.inv_entropy_tau)
                # calc per_sample_entropy
                prob, log_prob = logits.softmax(dim=-1), logits.log_softmax(dim=-1)  # both are (N, vocab_size)
                per_sample_entropy = torch.mean((-prob * log_prob).sum(dim=-1))
                # calc codebook_entropy
                avg_prob = prob.mean(dim=0)  # (vocab_size,)
                log_avg_prob = torch.log(avg_prob + 1e-7)
                codebook_entropy = (-avg_prob * log_avg_prob).sum()
                # calc entropy_loss
                entropy_loss = per_sample_entropy - codebook_entropy
            else:
                NxC_no_grad = f_BChw.detach().permute(0, 2, 3, 1).reshape(-1, C)
                NxC_no_grad = F.normalize(NxC_no_grad, dim=-1)
                idx_N = torch.argmax(NxC_no_grad @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                entropy_loss = 0
        else:  # not self.vocab_norm
            NxC_no_grad = f_BChw.detach().permute(0, 2, 3, 1).reshape(-1, C)
            E_dist = NxC_no_grad.square().sum(dim=1, keepdim=True) + self.embedding.weight.data.square().sum(dim=1,
                                                                                                             keepdim=False)
            E_dist.addmm_(NxC_no_grad, self.embedding.weight.data.T, alpha=-2, beta=1)  # E_dist: N x vocab_size
            idx_N = torch.argmin(E_dist, dim=1)
            entropy_loss = 0

        prob_per_class_is_chosen = idx_N.bincount(minlength=self.vocab_size).float()
        handler = tdist.all_reduce(prob_per_class_is_chosen, async_op=True) if (
                self.training and dist.initialized()) else None

        # look up
        idx_Bhw = idx_N.view(B, h, w)
        fhat_BChw = self.quant_resi(self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous())

        # calc loss
        vq_loss = F.mse_loss(fhat_BChw.detach(), f_BChw).mul_(self.beta) + F.mse_loss(fhat_BChw, f_BChw.detach())
        fhat_BChw = (fhat_BChw.detach() - f_BChw.detach()).add_(f_BChw)

        # update vocab_usage
        if handler is not None:
            handler.wait()
        prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
        vocab_usage = (prob_per_class_is_chosen > 0.01 / self.vocab_size).float().mean().mul_(100)

        if self.vocab_usage_record_times == 0:
            self.vocab_usage.copy_(prob_per_class_is_chosen)
        elif self.vocab_usage_record_times < 100:
            self.vocab_usage.mul_(0.9).add_(prob_per_class_is_chosen, alpha=0.1)
        else:
            self.vocab_usage.mul_(0.99).add_(prob_per_class_is_chosen, alpha=0.01)
        self.vocab_usage_record_times += 1

        return fhat_BChw, vq_loss, entropy_loss, (vocab_usage if ret_usages else None)

    def f_to_idx(self, f_BChw: torch.Tensor) -> torch.LongTensor:
        f_BChw = f_BChw.float()
        B, C, h, w = f_BChw.shape
        with torch.cuda.amp.autocast(enabled=False):
            # find the nearest embedding
            query_NxC = f_BChw.detach().permute(0, 2, 3, 1).reshape(-1, C)
            if self.vocab_norm:
                query_NxC = F.normalize(query_NxC, dim=-1)
                idx_N = torch.argmax(query_NxC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                E_dist = torch.sum(query_NxC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False)
                E_dist.addmm_(query_NxC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(E_dist, dim=1)
        return idx_N.view(B, h, w)


class VectorQuantizerHybrid(nn.Module):
    def __init__(
            self, vocab_size: int, vocab_width: int, vocab_norm: bool, beta: float = 0.25, quant_resi=-0.5,
            using_entropy_loss=False, entropy_temp=0.01,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.vocab_width: int = vocab_width
        self.register_buffer('vocab_usage', torch.zeros(self.vocab_size))
        self.vocab_usage_record_times: int = 0

        self.vocab_norm: bool = vocab_norm
        # self.quant_resi = ResConv(self.vocab_width, quant_resi=quant_resi)
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_width)
        self.beta: float = beta

        self.using_entropy_loss, self.inv_entropy_tau = using_entropy_loss, 1 / entropy_temp
        if not self.vocab_norm:
            assert not self.using_entropy_loss, 'entropy loss without vocab norm is not supported'

    def init_vocab(self, eini: float):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            base = self.vocab_width ** -0.5
            base /= 36
            self.embedding.weight.data.uniform_(-abs(eini) * base, abs(eini) * base)

    def extra_repr(self) -> str:
        return f'beta={self.beta:g}'

    def forward(self, class_tokens, patch_tokens, ret_usages=False):
        class_tokens = class_tokens.float()
        patch_tokens = patch_tokens.float()

        B, L, C = class_tokens.shape
        B, C, H, W = patch_tokens.shape
        patch_tokens = patch_tokens.flatten(start_dim=2).permute(0, 2, 1)
        NxC = torch.cat((class_tokens, patch_tokens), dim=1).reshape(-1, C)
        if self.vocab_norm:
            if self.using_entropy_loss:
                # find the nearest neighbor
                NxC_no_grad = NxC.detach()
                NxC_no_grad = F.normalize(NxC_no_grad, dim=-1)
                idx_N = torch.argmax(NxC_no_grad @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                # get logits
                E_dist = NxC.square().sum(dim=1, keepdim=True) + self.embedding.weight.square().sum(dim=1,
                                                                                                    keepdim=False)
                E_dist.addmm_(NxC, self.embedding.weight.T, alpha=-2, beta=1)  # E_dist: (N, vocab_size)
                logits = -E_dist.float().mul_(self.inv_entropy_tau)
                # calc per_sample_entropy
                prob, log_prob = logits.softmax(dim=-1), logits.log_softmax(dim=-1)  # both are (N, vocab_size)
                per_sample_entropy = torch.mean((-prob * log_prob).sum(dim=-1))
                # calc codebook_entropy
                avg_prob = prob.mean(dim=0)  # (vocab_size,)
                log_avg_prob = torch.log(avg_prob + 1e-7)
                codebook_entropy = (-avg_prob * log_avg_prob).sum()
                # calc entropy_loss
                entropy_loss = per_sample_entropy - codebook_entropy
            else:
                NxC_no_grad = NxC.detach()
                NxC_no_grad = F.normalize(NxC_no_grad, dim=-1)
                idx_N = torch.argmax(NxC_no_grad @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                entropy_loss = 0
        else:  # not self.vocab_norm
            NxC_no_grad = NxC.detach()
            E_dist = NxC_no_grad.square().sum(dim=1, keepdim=True) + self.embedding.weight.data.square().sum(dim=1,
                                                                                                             keepdim=False)
            E_dist.addmm_(NxC_no_grad, self.embedding.weight.data.T, alpha=-2, beta=1)  # E_dist: N x vocab_size
            idx_N = torch.argmin(E_dist, dim=1)
            entropy_loss = 0

        prob_per_class_is_chosen = idx_N.bincount(minlength=self.vocab_size).float()
        handler = tdist.all_reduce(prob_per_class_is_chosen, async_op=True) if (
                self.training and dist.initialized()) else None

        # look up
        fhat = self.embedding(idx_N)

        # calc loss
        vq_loss = F.mse_loss(fhat.detach(), NxC).mul_(self.beta) + F.mse_loss(fhat, NxC.detach())
        fhat = (fhat.detach() - NxC.detach()).add_(NxC)

        # update vocab_usage
        if handler is not None:
            handler.wait()
        prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
        vocab_usage = (prob_per_class_is_chosen > 0.01 / self.vocab_size).float().mean().mul_(100)

        if self.vocab_usage_record_times == 0:
            self.vocab_usage.copy_(prob_per_class_is_chosen)
        elif self.vocab_usage_record_times < 100:
            self.vocab_usage.mul_(0.9).add_(prob_per_class_is_chosen, alpha=0.1)
        else:
            self.vocab_usage.mul_(0.99).add_(prob_per_class_is_chosen, alpha=0.01)
        self.vocab_usage_record_times += 1

        fhat = fhat.view(B, -1, C)
        fhat_class = fhat[:, :L, :]
        fhat_patch = fhat[:, L:, :].view(B, H, W, C).permute(0, 3, 1, 2)

        return fhat_class, fhat_patch, vq_loss, entropy_loss, (vocab_usage if ret_usages else None)

    def f_to_idx(self, class_tokens, patch_tokens) -> torch.LongTensor:
        B, L, C = class_tokens.shape
        B, C, H, W = patch_tokens.shape
        class_tokens = class_tokens.float()
        patch_tokens = patch_tokens.float()
        patch_tokens = patch_tokens.flatten(start_dim=2).permute(0, 2, 1)
        NxC = torch.cat((class_tokens, patch_tokens), dim=1).reshape(-1, C)
        with torch.cuda.amp.autocast(enabled=False):
            # find the nearest embedding
            if self.vocab_norm:
                NxC = F.normalize(NxC, dim=-1)
                idx_N = torch.argmax(NxC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                E_dist = torch.sum(NxC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(),
                                                                                  dim=1, keepdim=False)
                E_dist.addmm_(NxC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(E_dist, dim=1)
        return idx_N


class VectorQuantizerX(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            vocab_width: int,
            beta: float = 0.25,
            use_entropy_loss=False,
            entropy_temp=0.01,
    ):
        super().__init__()
        self.beta = beta
        self.vocab_size = vocab_size
        self.vocab_width = vocab_width
        self.vocab_usage_record_times: int = 0
        self.register_buffer('vocab_usage', torch.zeros(self.vocab_size))

        self.codebook = NormalizedEmbedding(self.vocab_size, self.vocab_width)

        self.use_entropy_loss = use_entropy_loss
        self.inv_entropy_tau = 1 / entropy_temp

    def init_vocab(self, eini: float):
        if eini > 0:
            nn.init.trunc_normal_(self.codebook.weight.data, std=eini)
        elif eini < 0:
            base = self.vocab_width ** -0.5
            base /= 36
            self.codebook.weight.data.uniform_(-abs(eini) * base, abs(eini) * base)

    def extra_repr(self) -> str:
        return f'beta={self.beta:g}'

    def forward(self, features):
        B, L, C = features.shape
        features = features.reshape(-1, C)
        features = F.normalize(features, dim=-1).float()
        codebook_embed = self.codebook.get_norm_weight()
        indices = torch.argmax(features.detach() @ codebook_embed.T, dim=1)
        entropy_loss = get_entropy_loss(features, codebook_embed, self.inv_entropy_tau) if self.use_entropy_loss else 0
        features_hat = self.codebook(indices)

        # calc loss
        vq_loss = F.mse_loss(features_hat.detach(), features).mul_(self.beta) + F.mse_loss(features_hat,
                                                                                           features.detach())
        features_hat = (features_hat.detach() - features.detach()).add_(features)

        # update vocab_usage
        prob_per_class_is_chosen = indices.bincount(minlength=self.vocab_size).float()
        handler = tdist.all_reduce(prob_per_class_is_chosen, async_op=True) if (
                self.training and dist.initialized()) else None
        if handler is not None:
            handler.wait()
        prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
        vocab_usage = (prob_per_class_is_chosen > 0.01 / self.vocab_size).float().mean().mul_(100)
        if self.vocab_usage_record_times == 0:
            self.vocab_usage.copy_(prob_per_class_is_chosen)
        elif self.vocab_usage_record_times < 100:
            self.vocab_usage.mul_(0.9).add_(prob_per_class_is_chosen, alpha=0.1)
        else:
            self.vocab_usage.mul_(0.99).add_(prob_per_class_is_chosen, alpha=0.01)
        self.vocab_usage_record_times += 1

        return features_hat.view(B, L, C), vq_loss, entropy_loss, vocab_usage

    def f_to_idx(self, features):
        B, L, C = features.shape
        features = features.reshape(-1, C)
        features = F.normalize(features, dim=-1).float()
        codebook_embed = self.codebook.get_norm_weight().float()
        indices = torch.argmax(features.detach() @ codebook_embed.T, dim=1)
        return indices.view(B, L)


class VectorQuantizerM(nn.Module):
    def __init__(
            self,
            vocab_size,
            vocab_width,
            beta=0.25,
            use_entropy_loss=False,
            entropy_temp=0.01,
            num_codebooks=16
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebooks = nn.ModuleList()
        for _ in range(num_codebooks):
            codebook = VectorQuantizerX(
                vocab_size=vocab_size // num_codebooks,
                vocab_width=vocab_width // num_codebooks,
                beta=beta,
                use_entropy_loss=use_entropy_loss,
                entropy_temp=entropy_temp,
            )
            self.codebooks.append(codebook)

    def init_vocab(self, eini: float):
        for codebook in self.codebooks:
            codebook.init_vocab(eini)

    def f_to_idx(self, features):
        indices = []
        chunk_size = features.shape[-1] // self.num_codebooks
        splited_features = features.split(chunk_size, dim=-1)
        for i, codebook in enumerate(self.codebooks):
            indices.append(codebook.f_to_idx(splited_features[i]))
        indices = torch.stack(indices, dim=1)
        return indices

    def idx_to_f(self, indices):
        assert indices.shape[1] == self.num_codebooks
        latent_features = []
        for i, codebook in enumerate(self.codebooks):
            sub_indices = indices[:, i].flatten(start_dim=1)
            latent_feature = codebook.codebook(sub_indices)
            latent_features.append(latent_feature)
        latent_features = torch.cat(latent_features, dim=-1)
        return latent_features

    def forward(self, features):
        latent_features = []
        global_vq_loss = 0.
        global_entropy_loss = 0.
        global_vocab_usage = 0.
        chunk_size = features.shape[-1] // self.num_codebooks
        splited_features = features.split(chunk_size, dim=-1)
        for i, codebook in enumerate(self.codebooks):
            latent_feature, vq_loss, entropy_loss, vocab_usage = codebook(splited_features[i])
            latent_features.append(latent_feature)
            global_vq_loss += vq_loss
            global_entropy_loss += entropy_loss
            global_vocab_usage += vocab_usage
        latent_features = torch.cat(latent_features, dim=-1)
        global_entropy_loss /= self.num_codebooks
        global_vq_loss /= self.num_codebooks
        global_vocab_usage /= self.num_codebooks
        return latent_features, global_vq_loss, global_entropy_loss, global_vocab_usage


class CausalAttention(nn.Module):
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

        x = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=True)

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
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
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


from functools import partial
from timm.models.layers import create_conv2d, get_norm_act_layer, get_norm_layer, make_divisible

class GeGluMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        act_layer = None,
        drop = 0.0,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer('layernorm'), eps=1e-6)
        self.norm = norm_layer(in_features)
        self.act = nn.GELU(approximate='tanh')
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x
