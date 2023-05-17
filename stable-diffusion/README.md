# Benchmark on Stable Diffusion

## Setup

### Environment
* Create Conda environment:

  ```shell
  conda env create -f environment.yaml
  conda acitvate sige-sd
  ```

* Install other dependencies:

  ```
  pip install git+https://github.com/zhijian-liu/torchprofile
  ```

* Install SIGE following [../README.md](../README.md#installation).

**Notice**: Currently Stable Diffusion benchmark only supports CUDA with FP32 precision.

### Models

Download the model from https://github.com/CompVis/stable-diffusion. We used sd-v1-4 for our experiments. Put the model in `pretrained`.

```shell
mkdir -p pretrained
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O pretrained/sd-v1-4.ckpt
```

## Get Started

### Image Inpainting

#### Quality Results 

* Original Stable Diffusion

  ```shell
  python run.py --prompt "a photograph of a horse on a grassland" \
  --output_path inpainting-original.png \
  --init_img assets/inpainting/original/0.png \
  --mask_path assets/inpainting/masks/0.npy --W 1024 --seed 36
  ```

* SIGE Stable Diffusion

  ```shell
  python run.py --prompt "a photograph of a horse on a grassland" \
  --output_path inpainting-sige.png \
  --init_img assets/inpainting/original/0.png \
  --mask_path assets/inpainting/masks/0.npy --W 1024 --seed 36 \
  --config_path configs/sige.yaml
  ```

The generated images `inpainting-original.png` and `inpainting-sige.png` should look very similar.

#### Efficiency Results

* Original Stable Diffusion

  ```shell
  python run.py --prompt "a photograph of a horse on a grassland" \
  --init_img assets/inpainting/original/0.png \
  --mask_path assets/inpainting/masks/0.npy --W 1024 --seed 36 \
  --mode profile_unet
  ```

* SIGE Stable Diffusion

  ```shell
  python run.py --prompt "a photograph of a horse on a grassland" \
  --init_img assets/inpainting/original/0.png \
  --mask_path assets/inpainting/masks/0.npy --W 1024 --seed 36 \
  --config_path configs/sige.yaml \
  --mode profile_unet
  ```

You can also profile the decoder with the argument `--mode profile_decoder`. Reference results on NVIDIA RTX 3090:

<table>
<thead>
  <tr>
    <th rowspan="2" style="text-align: center;">Method</th>
    <th colspan="2" style="text-align: center;">UNet</th>
    <th colspan="2" style="text-align: center;">Decoder</th>
  </tr>
  <tr>
    <th style="text-align: center;">MACs (G)</th>
    <th style="text-align: center;">Latency (ms)</th>
    <th style="text-align: center;">MACs (G)</th>
    <th style="text-align: center;">Latency (ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align: center;">Original</td>
    <td style="text-align: center;">1854.8</td>
    <td style="text-align: center;">368.6</td>
    <td style="text-align: center;">2550.8</td>
    <td style="text-align: center;">235.0</td>
  </tr>
  <tr>
    <td style="text-align: center;">SIGE</td>
    <td style="text-align: center;">514.5</td>
    <td style="text-align: center;">95.0</td>
    <td style="text-align: center;">343.5</td>
    <td style="text-align: center;">48.0</td>
  </tr>
</tbody>
</table>

### Image-to-image translation

#### Quality Results

* Original Stable Diffusion

  ```shell
  # Example 0
  python run.py --task sdedit \
  --prompt "A fantasy landscape, trending on artstation" \
  --output_path img2img-original0.png \
  --edited_img assets/img2img/edited/0.png --seed 11
  
  # Example 1
  python run.py --task sdedit \
  --prompt "A fantasy beach landscape, trending on artstation" \
  --output_path img2img-original1.png \
  --edited_img assets/img2img/edited/1.png --seed 95
  ```

* SIGE Stable Diffusion

  ```shell
  # Example 0
  python run.py --task sdedit \
  --prompt "A fantasy landscape, trending on artstation" \
  --output_path img2img-sige0.png \
  --init_img assets/img2img/original/0.png \
  --edited_img assets/img2img/edited/0.png --seed 11 \
  --config_path configs/sige.yaml
  
  # Example 1
  python run.py --task sdedit \
  --prompt "A fantasy beach landscape, trending on artstation" \
  --output_path img2img-sige1.png \
  --init_img assets/img2img/original/1.png \
  --edited_img assets/img2img/edited/1.png --seed 95 \
  --config_path configs/sige.yaml
  ```

The generated images of each example should look very similar.

#### Efficiency Results

* Original Stable Diffusion

  ```shell
  # Example 0
  python run.py --task sdedit \
  --prompt "A fantasy landscape, trending on artstation" \
  --edited_img assets/img2img/edited/0.png --seed 11 --mode profile_unet
  
  # Example 1
  python run.py --task sdedit \
  --prompt "A fantasy beach landscape, trending on artstation" \
  --edited_img assets/img2img/edited/1.png --seed 95 --mode profile_unet
  ```

* SIGE Stable Diffusion

  ```shell
  # Example 0
  python run.py --task sdedit \
  --prompt "A fantasy landscape, trending on artstation" \
  --init_img assets/img2img/original/0.png \
  --edited_img assets/img2img/edited/0.png --seed 11 \
  --config_path configs/sige.yaml --mode profile_unet
  
  # Example 1
  python run.py --task sdedit \
  --prompt "A fantasy beach landscape, trending on artstation" \
  --init_img assets/img2img/original/1.png \
  --edited_img assets/img2img/edited/1.png --seed 95 \
  --config_path configs/sige.yaml --mode profile_unet
  ```

You can also profile the encoder with the argument `--mode profile_encoder` and decoder with `--mode profile_decoder`. Reference results on NVIDIA RTX 3090:

<table>
<thead>
  <tr>
    <th rowspan="2" style="text-align: center;">Method</th>
    <th rowspan="2" style="text-align: center;">Example</th>
    <th colspan="2" style="text-align: center;">UNet</th>
    <th colspan="2" style="text-align: center;">Eecoder</th>
    <th colspan="2" style="text-align: center;">Decoder</th>
  </tr>
  <tr>
    <th style="text-align: center;">MACs (G)</th>
    <th style="text-align: center;">Latency (ms)</th>
    <th style="text-align: center;">MACs (G)</th>
    <th style="text-align: center;">Latency (ms)</th>
    <th style="text-align: center;">MACs (G)</th>
    <th style="text-align: center;">Latency (ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align: center;">Original</td>
    <td style="text-align: center;">--</td>
    <td style="text-align: center;">1854.8</td>
    <td style="text-align: center;">368.6</td>
    <td style="text-align: center;">1152.1</td>
    <td style="text-align: center;">115.2</td>
    <td style="text-align: center;">2550.8</td>
    <td style="text-align: center;">235.0</td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align: center;">SIGE</td>
    <td style="text-align: center;">0</td>
    <td style="text-align: center;">224.9</td>
    <td style="text-align: center;">51.2</td>
    <td style="text-align: center;">39.2</td>
    <td style="text-align: center;">10.1</td>
    <td style="text-align: center;">154.2</td>
    <td style="text-align: center;">30.7</td>
  </tr>
  <tr>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">352.6</td>
    <td style="text-align: center;">76.4</td>
    <td style="text-align: center;">76.3</td>
    <td style="text-align: center;">14.5</td>
    <td style="text-align: center;">318.3</td>
    <td style="text-align: center;">45.6</td>
  </tr>
</tbody>
</table>
