# threestudio-mvimg-gen

Direct multi-view images generation extension for threestudio. To use it, please install [threestudio](https://github.com/threestudio-project/threestudio) first and then install this extension in threestudio `custom` directory.

Currently, we only support [stable-zero123](https://huggingface.co/stabilityai/stable-zero123), and we will support more methods including MVDream, SyncDreamer, Wonder3D in the future.

## Installation
```
cd custom
git clone https://github.com/DSaurus/threestudio-mvimg-gen.git
```
If you want to download stable-zero123 model, please go to `load/zero123` directory and run `download.sh`.

## Quick Start
```
python launch.py --config custom/threestudio-mvimg-gen/configs/stable-zero123.yaml --train --gpu 0 data.image_path=./load/images/catstatue_rgba.png
```

## Camera parameters in config file
```
  random_camera:
    # ------------------------------
    eval_elevation_deg: 0.0
    eval_camera_distance: 3.8
    eval_fovy_deg: 20.0
    n_test_views: 16
    # ------------------------------
```

## stable-zero123 parameters in config file
```
  guidance:
    pretrained_config: "./load/zero123/sd-objaverse-finetune-c_concat-256.yaml"
    pretrained_model_name_or_path: "./load/zero123/stable_zero123.ckpt"
    vram_O: false
    cond_image_path: ${data.image_path}
    cond_elevation_deg: ${data.default_elevation_deg}
    cond_azimuth_deg: ${data.default_azimuth_deg}
    cond_camera_distance: ${data.default_camera_distance}
    guidance_scale: 7.5
    min_step_percent: 0.98
    max_step_percent: 0.98

  num_inference_steps: 100
```
