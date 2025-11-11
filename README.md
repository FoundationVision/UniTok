<div align="center">
<img src="./assets/logo.png" alt="UniTok" width="200">

<h1>UniTok: A Unified Tokenizer <br> for Visual Generation and Understanding</h1>

[**Chuofan Ma**](https://machuofan.github.io/)<sup>1,2</sup> · [**Yi Jiang**](https://enjoyyi.github.io/)<sup>2&dagger;</sup> · [**Junfeng Wu**](https://wjf5203.github.io/)<sup>2,3</sup> · [**Jihan Yang**](https://jihanyang.github.io/)<sup>1</sup>
<br>
[**Xin Yu**](https://xinyu-andy.github.io/)<sup>1</sup> · [**Zehuan Yuan**](https://shallowyuan.github.io/)<sup>2*</sup> · [**Bingyue Peng**](https://openreview.net/profile?id=~BINGYUE_PENG1)<sup>2</sup> · [**Xiaojuan Qi**](https://xjqi.github.io/)<sup>1&dagger;*</sup>

<sup>1</sup>HKU&emsp;&emsp;&emsp;<sup>2</sup>ByteDance&emsp;&emsp;&emsp;<sup>3</sup>HUST
<br>
&dagger;project lead&emsp;&emsp;&emsp;*corresponding author

<a href="https://arxiv.org/abs/2502.20321"><img src='https://img.shields.io/badge/arXiv-UniTok-red' alt='Paper PDF'></a>
<a href="https://foundationvision.github.io/UniTok/"><img src='https://img.shields.io/badge/Project_Page-UniTok-green' alt='Project Page'></a>
<a href="https://huggingface.co/FoundationVision/unitok_tokenizer"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href="https://huggingface.co/spaces/FoundationVision/UniTok"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow'></a>

[//]: # (<a href='https://huggingface.co/datasets/depth-anything/DA-2K'><img src='https://img.shields.io/badge/Benchmark-DA--2K-yellow' alt='Benchmark'></a>)
</div>

This repo implements UniTok, a unified visual tokenizer well-suited for both generation and understanding tasks. 
It is compatiable with autoregressive generative models (e.g. LlamaGen), 
multimodal understanding models (e.g. LLaVA), and unified MLLMs (e.g. Chameleon and Liquid).

![teaser](assets/teaser.png)

Built upon UniTok, we construct an MLLM capable of both multimodal generation and understanding
with the [Liquid](https://github.com/FoundationVision/Liquid/) framework,
which sets a new state-of-the-art among unified autoregressive MLLMs.

![teaser](assets/samples.png)

## News
**2025-09-18:** UniTok is accepted at NeurIPS 2025 as a spotlight.

**2025-05-19:** We find UniTok favors generation **without classifier-free-guidance** -- 
it reduces gFID (without cfg) from 14.6 to 2.51 on ImageNet 256x256 using LlamaGen-XXL as the generator.
Please refer to the updated [EVAL.md](eval/EVAL.md) for more details.

**2025-04-15:** The [gradio demo](https://huggingface.co/spaces/FoundationVision/UniTok) of UniTok MLLM is available on Huggingface now!

**2025-04-02:** A new [checkpoint](https://huggingface.co/FoundationVision/unitok_tokenizer/tree/main) 
of UniTok is released, which has better downstream task performance 
by replacing the causal attention projection layer with full attention.
The [model weights](https://huggingface.co/FoundationVision/unitok_mllm) 
of our unified MLLM are also available on Huggingface now!

**2025-02-28:** Paper, code, model, and [project page](https://foundationvision.github.io/UniTok/) for UniTok are all released.


## Performance

<table>
    <thead>
        <tr>
            <th>Method</th>
            <th>#Tokens</th>
            <th>rFID &darr;</th>
            <th>Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan="4"><i>VQVAE Model</i></td>
        </tr>
        <tr align="center">
            <td>VQ-GAN</td>
            <td>256</td>
            <td>4.98</td>
            <td>--</td>
        </tr>
        <tr align="center">
            <td>RQ-VAE</td>
            <td>256</td>
            <td>1.30</td>
            <td>--</td>
        </tr>
        <tr align="center">
            <td>VAR</td>
            <td>680</td>
            <td>0.90</td>
            <td>--</td>
        </tr>
        <tr>
            <td colspan="4"><i>CLIP Model</i></td>
        </tr>
        <tr align="center">
            <td>CLIP</td>
            <td>256</td>
            <td>--</td>
            <td>76.2</td>
        </tr>
        <tr align="center">
            <td>SigLIP</td>
            <td>256</td>
            <td>--</td>
            <td>80.5</td>
        </tr>
        <tr align="center">
            <td>ViTamin</td>
            <td>256</td>
            <td>--</td>
            <td>81.2</td>
        </tr>
        <tr>
            <td colspan="4"><i>Unified Model</i></td>
        </tr>
        <tr align="center">
            <td>TokenFlow &dagger;</td>
            <td>680</td>
            <td>1.37</td>
            <td>--</td>
        </tr>
        <tr align="center">
            <td>VILA-U &dagger;</td>
            <td>256</td>
            <td>1.80</td>
            <td>73.3</td>
        </tr>
        <tr align="center">
            <td>UniTok</td>
            <td>256</td>
            <td>0.41</td>
            <td>70.8</td>
        </tr>
        <tr align="center">
            <td>UniTok &dagger;</td>
            <td>256</td>
            <td>0.38</td>
            <td>78.6</td>
        </tr>
    </tbody>
</table>


&dagger; indicates the model uses pretrained CLIP weights for initialization. Although CLIP weight initialization boosts ImageNet zero-shot accuracy,
we notice that random initialization leads to better downstream understanding performance.
We thus release the model checkpoint of UniTok that is trained from scratch.




## Model Weights

|    Model     | Res. | #Token |        Code Shape         | rFID |  Checkpoint  |
|:------------:|:----:|:------:|:-------------------------:|:----:|:------------:|
| UniTok-Large | 256  |  256   | 16 $\times$ 16 $\times$ 8 | 0.41 | [Download](https://huggingface.co/FoundationVision/unitok_tokenizer/blob/main/unitok_tokenizer.pth) |


## Usage

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.3.1

### Installation

```bash
git clone https://github.com/FoundationVision/UniTok.git
cd UniTok
pip install -r requirements.txt
```

### Inference

Please download the [checkpoint](https://huggingface.co/FoundationVision/unitok_tokenizer) and fill in the `ckpt_path`.
```bash
python inference.py \
    --ckpt_path /path/to/unitok_tokenizer.pth \
    --src_img /path/to/test_img --rec_img /path/to/rec_img
```

### Training

- We train UniTok on [DataComp-1B](https://github.com/mlfoundations/datacomp). 
Please follow the [instructions](https://github.com/mlfoundations/datacomp?tab=readme-ov-file#downloading-datacomp-1b) to download and prepare the data.

- Download the [models](https://huggingface.co/FoundationVision/unitok_external) used for loss calculation and put them under `./external`.

- Download the [ImageNet validation set](https://www.image-net.org/) for zero-shot accuracy evaluation.

- Download the ImageNet 256$\times$256 [reference batch](https://huggingface.co/datasets/FoundationVision/imagenet_reference_batch) for FID evaluation.

Configure `nnodes, nproc_per_node, node_rank, master_addr, master_port` in `launch.sh` and run:

```bash
bash launch.sh \
    --output_dir '/path/to/save/checkpoints/' \
    --train_data '/path/to/datacomp/shards/{00000000..00140146}.tar' \
    --imagenet_val '/path/to/imagenet_val/' \
    --fid_eval_src '/path/to/imagenet_reference_batch' \
    --fid_eval_dst '/path/to/save/imagenet_reconstructed_batch'
```
**Note:** For more hyper-parameter configurations, please check `utils/config.py`.

### Unified MLLM
We show that UniTok significantly boosts the performance of unified MLLMs.

Visual Understanding Performance on VQA Benchmarks.

|   Method   |      LLM       |  Res.   |  VQAv2   |   GQA    | TextVQA  |   POPE   |   MME    |  MM-Vet  |
|:----------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|   Show-o   |  Phi-1.5-1.3B  |   256   |   59.3   |   48.7   |    -     |   73.8   |   948    |    -     |
|   Liquid   |    Gemma-7B    |   512   |   71.3   |   58.4   |   42.4   |   81.1   |   1119   |    -     |
|   VILA-U   |   Llama-2-7B   |   256   |   75.3   |   58.3   |   48.3   |   83.9   |   1336   |   27.7   |
| **UniTok** | **Llama-2-7B** | **256** | **76.8** | **61.1** | **51.6** | **83.2** | **1448** | **33.9** |

Visual Generation Performance on GenAI-Bench.

<table>
    <thead>
    <tr>
        <th rowspan="2">Method</th>
        <th rowspan="2">Type</th>
        <th rowspan="2">Count</th>
        <th rowspan="2">Differ</th>
        <th rowspan="2">Compare</th>
        <th colspan="2">Logical</th>
        <th rowspan="2">Overall</th>
    </tr>
    <tr>
        <th>Negate</th>
        <th>Universal</th>
    </tr>
    </thead>
    <tbody>
    <tr align="center">
        <td>Show-o</td>
        <td>Discrete Diff.</td>
        <td>0.70</td>
        <td>0.62</td>
        <td>0.71</td>
        <td>0.51</td>
        <td>0.65</td>
        <td>0.60</td>
    </tr>
    <tr align="center">
        <td>VILA-U</td>
        <td>Autoregressive</td>
        <td>0.70</td>
        <td>0.71</td>
        <td>0.74</td>
        <td>0.53</td>
        <td>0.66</td>
        <td>0.64</td>
    </tr>
    <tr align="center">
        <td>Liquid</td>
        <td>Autoregressive</td>
        <td>0.76</td>
        <td>0.73</td>
        <td>0.74</td>
        <td>0.46</td>
        <td>0.74</td>
        <td>0.65</td>
    </tr>
    <tr align="center">
        <th>UniTok</th>
        <th>Autoregressive</th>
        <th>0.76</th>
        <th>0.79</th>
        <th>0.74</th>
        <th>0.46</th>
        <th>0.73</th>
        <th>0.67</th>
    </tr>
    </tbody>
</table>

Please refer to [EVAL.md](eval/EVAL.md) for more details.

### Evaluation

We also benchmark UniTok in terms of both understanding performance using the [LLaVA](https://github.com/haotian-liu/LLaVA) framework 
and generation performance using the [LLamaGen](https://github.com/FoundationVision/LlamaGen) framework.
Please refer to [EVAL.md](eval/EVAL.md) for more details.



## Acknowledgement
UniTok is built upon the awesome works
[VAR](https://github.com/FoundationVision/VAR),
[DataComp](https://github.com/mlfoundations/datacomp),
[Liquid](https://github.com/FoundationVision/Liquid/),
[LLaVA](https://github.com/haotian-liu/LLaVA/),
[LlamaGen](https://github.com/FoundationVision/LlamaGen/),
and [ViTamin](https://github.com/Beckschen/ViTamin).


## LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{unitok,
  title={UniTok: A Unified Tokenizer for Visual Generation and Understanding},
  author={Ma, Chuofan and Jiang, Yi and Wu, Junfeng and Yang, Jihan and Yu, Xin and Yuan, Zehuan and Peng, Bingyue and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2502.20321},
  year={2025}
}
```
