# Spatially Incremental Generative Engine (SIGE)

### [Paper](https://arxiv.org/abs/2211.02048) | [Project](https://www.cs.cmu.edu/~sige/) | [Slides](https://www.cs.cmu.edu/~sige/resources/slides.key) | [YouTube](https://youtu.be/rDPotGoPPkQ)

**[NEW!]** SIGE is accepted by T-PAMI!

**[NEW!]** SIGE supports [Stable Diffusion](./stable_diffusion) and Mac MPS backend! We also release the codes of [an interactive demo for DDPM](diffusion_demo) on M1 Macbook Pro!

**[NEW!]** SIGE is accepted by NeurIPS 2022! Our code and benchmark datasets are publicly available!

![teaser](https://github.com/lmxyy/sige/raw/main/assets/teaser.jpg)
*We introduce Spatially Incremental Generative Engine (SIGE),an engine that selectively performs computations at the edited regions for image editing applications. The computation and latency are measured for a single forward. For the above examples, SIGE significantly reduces the computation of [SDEdit](https://github.com/ermongroup/SDEdit) with [DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch) (4-6x) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion) (8x), and [GauGAN](https://github.com/NVlabs/SPADE) (15x) while preserving the image quality. When combined with existing model compression methods such as [GAN Compression](https://github.com/mit-han-lab/gan-compression), it further reduces the computation of GauGAN by 47x. On NVIDIA RTX 3090, SIGE achieves up to 7.2x speedups.*

Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models</br>
[Muyang Li](https://lmxyy.me/), [Ji Lin](http://linji.me/), [Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), [Song Han](https://songhan.mit.edu/), and [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)</br>
CMU, MIT, and Stanford</br>
In NeurIPS 2022.

## Demo

<p align="center">
  <img src="https://github.com/lmxyy/sige/raw/main/diffusion_demo/assets/demo.gif" width=600>
</p>


SIGE achieves 2x less conversion time compared to original DDPM on M1 MacBook Pro GPU as we selectively perform computation on the edited regions.

## Overview

![overview](https://github.com/lmxyy/sige/raw/main/assets/method.gif)*Tiling-based sparse convolution overview. For each convolution <i>F<sub>l</sub></i> in the network, we wrap it into SIGE Conv<sub><i>l</i></sub>. The activations of the original image are already pre-computed. When getting the edited image, we first compute a difference mask between the original and edited image and reduce the mask to the active block indices to locate the edited regions. In each SIGE Conv<sub><i>l</i></sub>, we directly gather the active blocks from the edited activation <i>A<sub>l</sub></i><sup>edited</sup> according to the reduced indices, stack the blocks along the batch dimension, and feed them into <i>F<sub>l</sub></i>. The gathered blocks have an overlap of width 2 if <i>F<sub>l</sub></i> is 3×3 convolution with stride 1. After getting the output blocks from <i>F<sub>l</sub></i>, we scatter them back into <i>F<sub>l</sub></i>(<i>A<sub>l</sub></i><sup>original</sup>) to get the edited output, which approximates <i>F<sub>l</sub></i>(<i>A<sub>l</sub></i><sup>edited</sup>).*

## Performance

### Efficiency

![overview](https://github.com/lmxyy/sige/raw/main/assets/results.jpg)
*With 1.2% edits, SIGE could reduce the computation of DDPM, Progressive Distillation and GauGAN by 7-18x, achieve a 2-4x speedup on NVIDIA RTX 3090, 3-5x speedup on Apple M1 Pro GPU and 4-14x on M1 Pro CPU. When combined with GAN Compression, it further reduces 50x computation on GauGAN, achieving 38x speedup on M1 Pro CPU. Please check our paper for more details and results.*

### Quality

![overview](https://github.com/lmxyy/sige/raw/main/assets/quality.jpg)*Qualitative results under different edit sizes. PD is Progressive Distillation. Our method well preserves the visual fidelity of the original model without losing global context.*

![quality-stable-diffusion](https://github.com/lmxyy/sige/raw/main/assets/quality-stable-diffusion.jpg)

*More qualitative results of Stable Diffusion on both image inpainting and editing, measured on NVIDIA RTX 3090.*

References:

* Denoising Diffusion Probabilistic Models (DDPM), Ho et al., ICLR 2020
* Denoising Diffusion Implicit Model (DDIM), Song et al., ICLR 2021
* Progressive Distillation for Fast Sampling of Diffusion Models, Salimans et al., ICLR 2022
* Semantic Image Synthesis with Spatially-Adaptive Normalization (GauGAN), Park et al., CVPR 2019
* GAN Compression: Efficient Architectures for Interactive Conditional GANs, Li et al., CVPR 2020
* High-Resolution Image Synthesis with Latent Diffusion Models, Rombach et al., CVPR 2022

## Prerequisites

* Python3
* CPU, M1 GPU, or NVIDIA GPU + CUDA CuDNN
* [PyTorch](https://pytorch.org) >= 1.7. For M1 GPU support, please install [PyTorch](https://pytorch.org)>=2.0.

## Getting Started

### Installation

After installing [PyTorch](https://pytorch.org), you should be able to install SIGE with PyPI

```shell
pip install sige
```

or via GitHub:

```shell
pip install git+https://github.com/lmxyy/sige.git
```

or locally for development

```shell
git clone git@github.com:lmxyy/sige.git
cd sige
pip install -e .
```

For MPS backend, please set the environment variables:

```shell
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Usage Example

See [example.py](https://github.com/lmxyy/sige/tree/main/example.py) for the minimal SIGE convolution example. Please first install SIGE with the above instructions and [torchprofile](https://github.com/zhijian-liu/torchprofile) with

```shell
pip install torchprofile
```

Then you can run it with

```shell
python example.py
```

We also have [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmxyy/sige/blob/main/example.ipynb) example.

### Benchmark

To reproduce the results of [DDPM](https://github.com/ermongroup/ddim) and [Progressive Distillation](https://github.com/google-research/google-research/tree/master/diffusion_distillation) or download the LSUN Church editing datasets, please follow the instructions in [diffusion/README.md](https://github.com/lmxyy/sige/tree/main/diffusion/README.md).

To reproduce the results of [GauGAN](https://github.com/NVlabs/SPADE) and [GAN Compression](https://github.com/mit-han-lab/gan-compression) or download the Cityscapes editing datasets, please follow the instructions in [gaugan/README.md](https://github.com/lmxyy/sige/tree/main/gaugan/README.md).

## Citation

If you use this code for your research, please cite our paper.

```bibtex
@inproceedings{li2022efficient,
  title={Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models},
  author={Li, Muyang and Lin, Ji and Meng, Chenlin and Ermon, Stefano and Han, Song and Zhu, Jun-Yan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

## Acknowledgments

Our code is developed based on [SDEdit](https://github.com/ermongroup/SDEdit), [ddim](https://github.com/ermongroup/ddim), [diffusion_distillation](https://github.com/google-research/google-research/tree/master/diffusion_distillation), [gan-compression](https://github.com/mit-han-lab/gan-compression), [dpm-solver](https://github.com/LuChengTHU/dpm-solver), and [stable-diffusion](https://github.com/CompVis/stable-diffusion). We refer to [sbnet](https://github.com/uber-research/sbnet) for the tiling-based sparse convolution algorithm implementation. Our work is also inspired by the gather/scatter implementations in [torchsparse](https://github.com/mit-han-lab/torchsparse).

We thank [torchprofile](https://github.com/zhijian-liu/torchprofile) for MACs measurement, [clean-fid](https://github.com/GaParmar/clean-fid) for FID computation and [drn](https://github.com/fyu/drn) for Cityscapes mIoU computation.

We thank Yaoyao Ding, Zihao Ye, Lianmin Zheng, Haotian Tang, and Ligeng Zhu for the helpful comments on the engine design. We also thank George Cazenavette, Kangle Deng, Ruihan Gao, Daohan Lu, Sheng-Yu Wang and Bingliang Zhang for their valuable feedback. The project is partly supported by NSF, MIT-IBM Watson AI Lab, Kwai Inc, and Sony Corporation.
