import os
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm

from streamvggt.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from streamvggt.datasets.utils.image import imread_cv2

# preserves the original DUSt3R HAMMER default; override via the constructor or
# the DatasetConfig CLI rather than editing this constant.
DEFAULT_MAX_INTERVAL = 20


class HAMMER_Multi(BaseMultiViewDataset):
    """HAMMER RGB-D video sequences (https://github.com/Junggy/HAMMER-dataset),
    polarization (RGB) camera with rendered ground-truth depth, processed by
    datasets_preprocess/preprocess_hammer.py into:
        ROOT/<split>/<sequence>/{rgb,depth,cam}/XXXXXX.* + scene_metadata.npz
    Depth PNGs are uint16 millimeters; poses are metric cam2world."""

    def __init__(
        self,
        *args,
        ROOT,
        max_interval=DEFAULT_MAX_INTERVAL,
        is_metric=True,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = is_metric
        if not isinstance(max_interval, int) or max_interval < 1:
            raise ValueError(
                f"HAMMER max_interval must be a positive int, got {max_interval!r}"
            )
        self.max_interval = max_interval
        super().__init__(*args, **kwargs)
        if self.split not in ("train", "test"):
            raise ValueError(
                f"HAMMER split must be 'train' or 'test', got {self.split!r}"
            )
        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        self.scene_root = osp.join(self.ROOT, split)
        if not osp.isdir(self.scene_root):
            raise FileNotFoundError(
                f"HAMMER split directory not found: {self.scene_root} "
                f"(run datasets_preprocess/preprocess_hammer.py first)"
            )
        self.scenes = sorted(
            scene
            for scene in os.listdir(self.scene_root)
            if scene.startswith("scene")
        )
        if not self.scenes:
            raise ValueError(f"No HAMMER sequences found in {self.scene_root}")

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.scene_root, scene)
            with np.load(osp.join(scene_dir, "scene_metadata.npz")) as data:
                basenames = list(data["images"])
            num_imgs = len(basenames)
            # fail fast on a partial/corrupt preprocessing run instead of
            # erroring frames-deep into training
            for subdir, ext in (("rgb", ".png"), ("depth", ".png"), ("cam", ".npz")):
                files = [
                    f
                    for f in os.listdir(osp.join(scene_dir, subdir))
                    if f.endswith(ext)
                ]
                if len(files) != num_imgs:
                    raise ValueError(
                        f"{scene}: {len(files)} files in {subdir}/ but "
                        f"{num_imgs} frames in scene_metadata.npz"
                    )

            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            if num_imgs < cut_off:
                print(f"Skipping {scene}: only {num_imgs} frames < {cut_off} views")
                continue

            start_img_ids.extend(img_ids[: num_imgs - cut_off + 1])
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            offset += num_imgs
            j += 1

        if not scenes:
            raise ValueError(
                f"All HAMMER sequences in {self.scene_root} are shorter than "
                f"{self.num_views} views"
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
            rng,
            max_interval=self.max_interval,
            video_prob=0.6,
            fix_interval_prob=0.6,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.scene_root, self.scenes[scene_id])

            basename = self.images[view_idx]

            rgb_image = imread_cv2(osp.join(scene_dir, "rgb", basename + ".png"))
            depthmap = imread_cv2(
                osp.join(scene_dir, "depth", basename + ".png"), cv2.IMREAD_UNCHANGED
            )
            depthmap = depthmap.astype(np.float32) / 1000.0
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            cam = np.load(osp.join(scene_dir, "cam", basename + ".npz"))
            camera_pose = cam["pose"]
            intrinsics = cam["intrinsics"]
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="hammer",
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
        assert len(views) == num_views
        return views
