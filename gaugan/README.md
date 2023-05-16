# Benchmark on GauGAN

## Setup

### Environment

* Create Conda environment:

  ```shell
  conda create -n sige python=3
  conda activate sige
  ```

  For Apple M1 Pro CPU results, we used [Intel Anaconda](https://repo.anaconda.com/archive/Anaconda3-2022.10-MacOSX-x86_64.pkg).

* Install [PyTorch](https://pytorch.org). To reproduce our CUDA and CPU results, please use PyTorch 1.7. To enable MPS backend, please install PyTorch>=2.0.

* Install other dependencies:

  ```shell
  conda install pandas autopep8
  conda install tqdm -c conda-forge
  pip install blobfile torchprofile pyyaml lmdb clean-fid opencv-python gdown easydict dominate scikit-image lpips dominate
  ```

* Install SIGE following [../README.md](../README.md#installation).

### Dataset

You could download the dataset with

```shell
python download_dataset.py
```

The dataset will be stored at `database/cityscapes-edit`. Thre directory structure is like:

```text
database
└── cityscapes-edit
    ├── gt_instances
    ├── gt_labels
    ├── images
    ├── masks
    ├── meta.csv
    ├── synthetic_instances
    └── synthetic_labels
```

Specifically,

* `meta.csv` is a meta file that matches the synthetic semantic maps and the ground-truth semantic maps.
* `gt_instances`, `gt_labels` and `images` store the ground-truth instance labels, semantic labels and images, respectively.
* `synthetic_instances` and `synthetic_labels` store the synthetic instance labels and semantic labels, respectively.
* `masks` stores the difference masks between the synthetic semantic maps and the corresponding ground-truth semantic maps in `.npy` format.

## Get Started

We provide [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmxyy/sige/blob/main/gaugan/gaugan.ipynb) to quickly test GauGAN model with SIGE.

### Quality Results

* Generate images:

  ```shell
  # Original GauGAN
  python test.py --netG spade --use_pretrained \
  --save_dir results/gaugan
  
  # GuaGAN with SIGE
  python test.py \
  --netG sige_fused_spade --use_pretrained \
  --save_dir results/sige_gaugan --dont_save_label 
  
  # GAN Compression
  python test.py --netG sub_mobile_spade --use_pretrained \
  --save_dir results/gan-compression --dont_save_label \
  --config_str 32_32_32_48_32_24_24_32
  
  # GAN Compression with SIGE
  python test.py --netG sige_fused_sub_mobile_spade \
  --save_dir results/sige_fused_gc --dont_save_label \
  --config_str 32_32_32_48_32_24_24_32 --use_pretrained \
  --num_sparse_layers 4
  ```

  Note:
  
  * The results will be saved in the directory specified with `--save_dir`. For the generated results, `$id_synthetic.png` means using the synthetic map `$id` as the edited input and the corresponding ground-truth map as the original input. `$id_gt.png` means using the synthetic map `$id` as the original input and the corresponding ground-truth map as the edited input.
  * You could use `--dont_save_label` option to skip saving the input semantic maps in your results.
  
* Metric Measurement:

  You could use the script `get_metric.py` to measure the PSNR, LPIPS, FID and mIoU of your generated images. Specifically,

  * PSNR/LPIPS:

    ```shell
    # PSNR
    python get_metric.py --metric psnr --root $PATH_TO_YOUR_GENERATED_IMAGES
    
    # LPIPS
    python get_metric.py --metric lpips --root $PATH_TO_YOUR_GENERATED_IMAGES
    ```

    Note:

    * By default, these commands will compute the metrics against the ground-truth images. If you want to compute the metrics against the images of the original model, please specify the directory of the original model results with `--ref_root` and `--mode original`.
    * If you want to compute the metrics only at the edited regions, you could specify the mask root with `--mask_root database/cityscapes-edit/masks`.

  * FID:

    ```shell
    python get_metric.py --metric fid --root $PATH_TO_YOUR_GENERATED_IMAGES
    ```

  * mIoU:

    ```shell
    python get_metric.py --metric miou --image_root $PATH_TO_YOUR_GENERATED_IMAGES
    ```

    You could also specify the mask root with `--mask_root database/cityscapes-edit/masks` to compute the mIoU only at the edited regions.

### Efficiency Results

```shell
# Original GauGAN
python test.py --netG spade --mode profile --image_ids 1501 --no_symmetric_editing

# GuaGAN with SIGE
python test.py --netG sige_fused_spade --mode profile --image_ids 1501 --no_symmetric_editing

# GAN Compression
python test.py --netG sub_mobile_spade --mode profile \
--config_str 32_32_32_48_32_24_24_32 --image_ids 1501 --no_symmetric_editing

# GAN Compression with SIGE
python test.py --netG sige_fused_sub_mobile_spade --mode profile \
--config_str 32_32_32_48_32_24_24_32 --num_sparse_layers 4 \
--image_ids 1501 --no_symmetric_editing
```

Note:

* By default, these commands will test results on GPU. For CPU results, you could specify `--device cpu`.

* You could specify the test editing sample with `--image_ids`. It also support multiple samples, sperated by white space.

* You could change the number of the warmup and test rounds with `--warmup_times` and `--test_times`.

* You could disable the symmetric_editing with `--no_symmetric_editing` to speedup the measurement. For a ground-truth semantic map and synthetic map pair, these two kinds of editing are symmetric:

  * View the groud-truth map as the original input and the synthetic map as the edited input.
  * View the synthetic map as the original input and the ground-truth map as the edited input.

  As the symmetric editing shares the same editing regions, so it wil not affect the efficiency results.
