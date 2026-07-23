import os
import os.path as osp

import cv2
import numpy as np

from .arkitscenes import DEFAULT_STRIDE_RANGE
from .base.base_multiview_dataset import (
    BaseMultiViewDataset,
    EmptyDatasetError,
    intrinsics_rows_to_K,
)
from .types import Split
from .utils.image import imread_cv2
from .utils.zipio import frames_root


class ARKitScenesHighRes_Multi(BaseMultiViewDataset):
    """ARKitScenes "upsampling" scenes with laser-scanner GT depth and
    per-frame trajectories, preprocessed into:
        ROOT/<Training|Validation>/<scene>/{vga_wide,highres_depth}/... +
        scene_metadata.npz

    Unlike the low-res variant this keeps the original DUSt3R high-res data
    contract: scenes are enumerated from the directory tree (no
    all_metadata.npz), per-scene metadata is the raw preprocess output
    scene_metadata.npz (no pairs / image_collection, hence no generate_set
    step), and view sequences are sampled from timestamp-ordered frames only.
    The low-res loader excludes every scene present in this tree, so the two
    variants partition ARKitScenes between them."""

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
                self.split_dir = "Training"
            case Split.TEST:
                self.split_dir = "Validation"
            case _:
                raise ValueError(
                    f"ARKitScenesHighRes split must be Split.TRAIN or "
                    f"Split.TEST, got {self.split!r}"
                )

        self.loaded_data = self._load_data(self.split_dir)

    def _load_data(self, split):
        all_scenes = sorted(
            [
                d
                for d in os.listdir(osp.join(self.ROOT, split))
                if osp.isdir(osp.join(self.ROOT, split, d))
            ]
        )
        offset = 0
        scenes = []
        sceneids = []
        images = []
        start_img_ids = []
        scene_img_list = []
        intrinsics = []
        trajectories = []
        scene_id = 0
        for scene in all_scenes:
            scene_dir = osp.join(self.ROOT, split, scene)
            with np.load(osp.join(scene_dir, "scene_metadata.npz")) as data:
                # order frames by NUMERIC timestamp ("{scene}_{t}.png"); a
                # lexicographic filename sort misorders any scene whose
                # timestamps cross a digit boundary (e.g. "98.764" sorts
                # after "101.663"), corrupting is_video sequence ordering
                imgs_with_indices = sorted(
                    enumerate(data["images"]),
                    key=lambda x: float(x[1].split("_")[1][:-4]),
                )
                imgs = [x[1] for x in imgs_with_indices]
                cut_off = self.min_views()
                if len(imgs) < cut_off:
                    print(f"Skipping {scene}")
                    continue
                indices = [x[0] for x in imgs_with_indices]
                if not all(img[:8] == scene for img in imgs):
                    raise RuntimeError(
                        f"ARKitScenesHighRes frame/scene mismatch in {scene}: "
                        f"{[img for img in imgs if img[:8] != scene][:3]}"
                    )
                num_imgs = len(imgs)
                img_ids = list(np.arange(num_imgs) + offset)
                start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                scenes.append(scene)
                scene_img_list.append(img_ids)
                sceneids.extend([scene_id] * num_imgs)
                images.extend(imgs)
                start_img_ids.extend(start_img_ids_)

                intrinsics.extend(
                    list(intrinsics_rows_to_K(data["intrinsics"][indices]))
                )
                trajectories.extend(list(data["trajectories"][indices]))

                # offset groups
                offset += num_imgs
                scene_id += 1

        if not scenes:
            # distinguishes a stub / partially-downloaded tree from a real one
            # here, instead of an IndexError at sampling time
            raise EmptyDatasetError(
                f"ARKitScenesHighRes found no usable scenes under "
                f"{osp.join(self.ROOT, split)}"
            )
        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list
        self.intrinsics = intrinsics
        self.trajectories = trajectories
        self.start_img_ids = start_img_ids
        if not (len(self.images) == len(self.intrinsics) == len(self.trajectories)):
            raise RuntimeError(
                f"ARKitScenesHighRes metadata length mismatch: "
                f"{len(self.images)} images, {len(self.intrinsics)} intrinsics, "
                f"{len(self.trajectories)} trajectories"
            )

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
            # frames.zip when present (inode-safe layout), else the scene dir
            scene_dir = frames_root(
                osp.join(self.ROOT, self.split_dir, self.scenes[scene_id])
            )

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]
            if basename[:8] != self.scenes[scene_id]:
                raise RuntimeError(
                    f"ARKitScenesHighRes frame/scene mismatch: basename "
                    f"{basename!r} does not belong to scene "
                    f"{self.scenes[scene_id]!r}"
                )
            # Load RGB image
            rgb_image = imread_cv2(
                osp.join(scene_dir, "vga_wide", basename.replace(".png", ".jpg"))
            )
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(scene_dir, "highres_depth", basename), cv2.IMREAD_UNCHANGED
            )
            depthmap = depthmap.astype(np.float32) / 1000.0
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.7, 0.25, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="arkitscenes_highres",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.99, dtype=np.float32),
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
                f"ARKitScenesHighRes produced {len(views)} views but "
                f"{num_views} were requested"
            )
        return views
