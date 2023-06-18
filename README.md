# C3VD
Loader for the [C3VD dataset](https://durrlab.github.io/C3VD/).


# Installation

### 0. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "tinycudann" to install dependencies and create an environment

### 1. Clone this repo
```git clone https://github.com/gong-xuan/nerfstudio-c3vd.git```

### 2. Install this repo as a python package
Navigate to this folder and run `python -m pip install -e .`

### 3. Run `ns-install-cli`

### Checking the install

Run `ns-train nerfacto c3vd-data -h`, and check if you see
```
usage: ns-train nerfacto c3vd-data [-h] [--data PATH] [--scale-factor FLOAT] [--alpha-color STR] [--downscale-factor INT] [--scene-box-bound FLOAT]

╭─ arguments ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.datamanager.dataparser arguments ──────────────────────────────────────────────────────────────────╮
│ --data PATH             Directory specifying location of data. (default: /root/data/colon3d/reg_videos/c1v1) │
│ --scale-factor FLOAT    How much to scale the camera origins by. (default: 100.0)                            │
│ --alpha-color STR       alpha color of background (default: white)                                           │
│ --downscale-factor INT  How much to downscale images. (default: 1)                                           │
│ --scene-box-bound FLOAT                                                                                      │
│                         Boundary of scene box. (default: 1.5)                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

# Using the dataloader

## Run
- Launch training with `ns-train nerfacto c3vd-data --data <data_folder>`. This specifies a data folder to use.
    - example: `ns-train nerfacto c3vd-data --data ~/data/colon3d/reg_videos/c1v1/`
- Connect to the viewer by forwarding the viewer port, and click the link to `viewer.nerf.studio` provided in the output of the train script
    - Tip: click one of the training images in the viewer to get a good viewing pose
