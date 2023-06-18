from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Type

import cv2
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
# from nerfstudio.utils.rich_utils import CONSOLE


def load_colon3d_data_trajectory(root_path, load_depth=False):
    pose_file = f"{root_path}/pose.txt"
    with open(pose_file) as f:
        lines = f.readlines()
    poses = []
    imgs = []
    depths = [] 
    for n, line in enumerate(lines):
        line = line.replace("\n", "").split(',')
        assert len(line)==16
        pose = np.array([float(l) for l in line]).reshape((4,4))
        poses.append(pose)
        # imgs.append(imageio.imread(f"{root_path}/{n}_color.png"))
        imgs.append(f"{root_path}/{n}_color.png")
        if load_depth:
            depth = 100*cv2.imread(f"{root_path}/{n}_depth.tiff",3)[:,:,0]/65535.
            depths.append(depth)
    # imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)?
    poses = np.array(poses).astype(np.float32)
    if load_depth:
        depths = np.array(depths).astype(np.float32)
        return imgs, poses, depths
    else:
        return imgs, poses, None


def generate_newposes():
    w2c = torch.tensor([[1,0,0,-50],
                        [0,1,0,-50],
                        [0,0,1,70],
                        [0,0,0,1]])
    
    def dx(r, theta):
        return r*np.cos(theta/180.*np.pi)
    def dy(r, theta):
        return r*np.sin(theta/180.*np.pi)  
    # return torch.stack([torch.tensor([[1,0,0, -dx(10., angle)],
    #                                 [0,1,0, -dy(10., angle)],
    #                                 [0,0,1,-70],
    #                                 [0,0,0,1]]) for angle in np.linspace(-180,180,320)], 0).float()
    return torch.stack([torch.tensor([[1,0,0, 50],
                                    [0,1,0, 50],
                                    [0,0,1, -70+dz],
                                    [0,0,0,1]]) for dz in np.linspace(-30,100,320)], 0).float()


def load_colon_data(K, datadir, load_depth=False):
    focal = 767.5 #TODO: where focal is used?
    imgs, poses, depths = load_colon3d_data_trajectory(datadir, load_depth=load_depth)
    H, W = cv2.imread(imgs[0]).shape[:2]
    N = len(imgs)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0) #TODO
    render_poses = generate_newposes()

    batch_size = poses.shape[0]
    intrinsics = np.repeat(K[None], batch_size, axis=0)
    
    #split
    total_indices = np.arange(N)
    i_train = total_indices[::2]
    i_valtest= total_indices[1:][::2]
    i_val= i_valtest[::2]
    i_test= i_valtest[1:][::2]
    assert len(set(total_indices)) == len(set(i_train.tolist()+i_val.tolist()+i_test.tolist()))

    return imgs, depths, poses, intrinsics,  render_poses, [H, W, focal], [i_train, i_val, i_test]


@dataclass
class C3VDDataParserConfig(DataParserConfig):
    """C3VD (https://arxiv.org/abs/2210.13445) dataset parser config"""

    _target: Type = field(default_factory=lambda: C3VD)
    """target class to instantiate"""
    data: Path = Path("~/data/colon3d/reg_videos/c1v1").expanduser()
    """Directory specifying location of data."""
    scale_factor: float = 100.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    downscale_factor: int = 1
    """How much to downscale images."""
    scene_box_bound: float = 1.5
    """Boundary of scene box."""


@dataclass
class C3VD(DataParser):
    """C3VD"""

    config: C3VDDataParserConfig

    def __init__(self, config: C3VDDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.K = np.asarray([[767.3861511125845, 0, 679.054265997005],[0,767.5058656118406,543.646891684636],[0,0,1]])
        self.D = np.asarray([[-0.18867185058223412],[-0.003927337093919806],[0.030524814153620117],[-0.012756926010904904]]).reshape([-1])
        self.D = torch.as_tensor([*self.D.tolist(),0,0])

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None
        splits_dir = self.data / "splits"

        image_filenames, depths, poses, intrinsics,  render_poses, [H, W, focal], [i_train, i_val, i_test] = load_colon_data(self.K, self.data, False)
        if split != 'train':
            image_filenames = image_filenames[::10]
            poses = poses[::10]
        cams = []
        center = poses[:,:3,3].mean()
        poses[:,:3,3] -= center
        poses[:,:3,3] /= self.scale_factor
        
        for pose in poses:
            # from opencv coord to opengl coord (used by nerfstudio)
            pose[0:3, 1:3] *= -1  # switch cam coord x,y
            pose = pose[[1, 0, 2], :]  # switch world x,y
            pose[2, :] *= -1  # invert world z
            # for aabb bbox usage
            pose = pose[[1, 2, 0], :]  # switch world xyz to zxy

            cams.append(
                {
                    "camera_to_worlds": pose,
                    "fx": self.K[0,0],
                    "fy": self.K[1,1],
                    "cx": self.K[0,2],
                    "cy": self.K[1,2],
                    "height": H,
                    "width": W,
                    "distortion_params": self.D,
                }
            )

        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-self.config.scene_box_bound] * 3, [self.config.scene_box_bound] * 3], dtype=torch.float32
            )
        )
        cam_dict = {}
        for k in cams[0].keys():
            cam_dict[k] = torch.stack([torch.as_tensor(c[k]) for c in cams], dim=0)
        cameras = Cameras(camera_type=CameraType.FISHEYE, **cam_dict)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
        )

        return dataparser_outputs

