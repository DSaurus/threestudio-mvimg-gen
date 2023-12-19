# threestudio-mvimg-gen

![mvimg-gen](https://github.com/DSaurus/threestudio-mvimg-gen/assets/24589363/c1feae91-b8f6-44ea-9790-fbcd9ccc2006)


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
