import os
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm

from .base.base_multiview_dataset import BaseMultiViewDataset, EmptyDatasetError
from .types import Split
from .utils.image import imread_cv2

# preserves the original DUSt3R ScanNet stride cap; override via the constructor
# or the DatasetConfig CLI rather than editing this constant.
DEFAULT_STRIDE_RANGE = (1, 30)


class ScanNet_Multi(BaseMultiViewDataset):
    """ScanNet RGB-D video sequences with metric depth and per-frame cam2world
    poses, preprocessed into:
        ROOT/{scans_train,scans_test}/<scene>/{color,depth,cam}/... +
        new_scene_metadata.npz."""

    def __init__(
        self,
        *args,
        ROOT,
        stride_range=DEFAULT_STRIDE_RANGE,
        is_metric=True,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = is_metric
        super().__init__(*args, stride_range=stride_range, **kwargs)
        match self.split:
            case Split.TRAIN:
                subdir = "scans_train"
            case Split.TEST:
                subdir = "scans_test"
            case _:
                raise ValueError(
                    f"ScanNet split must be Split.TRAIN or Split.TEST, got {self.split!r}"
                )

        self.loaded_data = self._load_data(subdir)

    def _load_data(self, subdir):
        self.scene_root = osp.join(self.ROOT, subdir)
        if not osp.isdir(self.scene_root):
            raise FileNotFoundError(
                f"ScanNet split directory not found: {self.scene_root}"
            )
        self.scenes = [
            scene for scene in os.listdir(self.scene_root) if scene.startswith("scene")
        ]
        if not self.scenes:
            raise ValueError(f"No ScanNet sequences found in {self.scene_root}")

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.scene_root, scene)
            with np.load(
                osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True
            ) as data:
                basenames = data["images"]
                num_imgs = len(basenames)
                img_ids = list(np.arange(num_imgs) + offset)
                cut_off = (
                    self.num_views
                    if not self.allow_repeat
                    else max(self.num_views // 3, 3)
                )
                start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                if num_imgs < cut_off:
                    print(f"Skipping {scene}")
                    continue

                start_img_ids.extend(start_img_ids_)
                sceneids.extend([j] * num_imgs)
                images.extend(basenames)
                scenes.append(scene)
                scene_img_list.append(img_ids)

                # offset groups
                offset += num_imgs
                j += 1

        if not scenes:
            raise EmptyDatasetError(
                f"ScanNet found no usable scenes under {self.scene_root}"
            )
        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,  # stride/order policy: self.stride_range + base defaults
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.scene_root, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "color")
            depth_dir = osp.join(scene_dir, "depth")
            cam_dir = osp.join(scene_dir, "cam")

            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(depth_dir, basename + ".png"), cv2.IMREAD_UNCHANGED
            )
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            camera_pose = cam["pose"]
            intrinsics = cam["intrinsics"]
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="ScanNet",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        if len(views) != num_views:
            raise RuntimeError(
                f"ScanNet produced {len(views)} views but {num_views} were requested"
            )
        return views
