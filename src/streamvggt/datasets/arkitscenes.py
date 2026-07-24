import os
import os.path as osp

import cv2
import numpy as np

from .base.base_multiview_dataset import (
    BaseMultiViewDataset,
    EmptyDatasetError,
    intrinsics_rows_to_K,
)
from .types import Split
from .utils.image import imread_cv2
from .utils.zipio import frames_root

# preserves the original DUSt3R ARKitScenes stride cap; override via the
# constructor or the DatasetConfig CLI rather than editing this constant.
DEFAULT_STRIDE_RANGE = (1, 8)


def stratified_sampling(indices, num_samples, rng=None):
    if num_samples > len(indices):
        raise ValueError("num_samples cannot exceed the number of available indices.")
    elif num_samples == len(indices):
        return indices

    sorted_indices = sorted(indices)
    stride = len(sorted_indices) / num_samples
    sampled_indices = []
    if rng is None:
        rng = np.random.default_rng()

    for i in range(num_samples):
        start = int(i * stride)
        end = int((i + 1) * stride)
        # Ensure end does not exceed the list
        end = min(end, len(sorted_indices))
        if start < end:
            # Randomly select within the current stratum
            rand_idx = rng.integers(start, end)
            sampled_indices.append(sorted_indices[rand_idx])
        else:
            # In case of any rounding issues, select the last index
            sampled_indices.append(sorted_indices[-1])

    return rng.permutation(sampled_indices)


class ARKitScenes_Multi(BaseMultiViewDataset):
    """ARKitScenes RGB-D video sequences with metric (lowres) depth and
    per-frame trajectories, preprocessed into:
        ROOT/<Training|Test>/<scene>/{vga_wide,lowres_depth}/... +
        new_scene_metadata.npz (and a sibling ROOT_highres/ tree whose scenes
        are excluded here so the high-res variant can own them)."""

    def __init__(
        self,
        *args,
        ROOT,
        stride_range=DEFAULT_STRIDE_RANGE,
        regular_stride=True,
        is_metric=True,
        highres_root=None,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = is_metric
        # explicit root of the highres sibling tree whose scenes this loader
        # must exclude; None falls back to the original DUSt3R convention of
        # deriving ROOT + "_highres" (silently skipped when absent)
        self.highres_root = highres_root
        super().__init__(
            *args, stride_range=stride_range, regular_stride=regular_stride, **kwargs
        )
        match self.split:
            case Split.TRAIN:
                self.split_dir = "Training"
            case Split.TEST:
                self.split_dir = "Test"
            case _:
                raise ValueError(
                    f"ARKitScenes split must be Split.TRAIN or Split.TEST, "
                    f"got {self.split!r}"
                )

        self.loaded_data = self._load_data(self.split_dir)

    def _load_data(self, split):
        with np.load(osp.join(self.ROOT, split, "all_metadata.npz")) as data:
            self.scenes: np.ndarray = data["scenes"]
            high_res_list = np.array([])
            # the highres tree uses Training/Validation (not Training/Test),
            # so resolve its subdir from the Split enum instead of reusing the
            # lowres directory name in `split`
            highres_dir = "Training" if self.split == Split.TRAIN else "Validation"
            if self.highres_root is not None:
                # explicit exclusion root (from the config, which knows the
                # real highres path): must exist -- a typo here would silently
                # double-count the highres scenes in both variants
                highres_split_dir = os.path.join(str(self.highres_root), highres_dir)
                if not os.path.isdir(highres_split_dir):
                    raise FileNotFoundError(
                        f"ARKitScenes highres_root was given explicitly but "
                        f"{highres_split_dir} does not exist"
                    )
                high_res_list = np.array(os.listdir(highres_split_dir))
            else:
                # original DUSt3R convention: sibling tree named ROOT_highres;
                # silently skipped when absent (lowres-only setups)
                highres_split_dir = os.path.join(
                    self.ROOT.rstrip("/") + "_highres",
                    highres_dir,
                )
                if os.path.isdir(highres_split_dir):
                    high_res_list = np.array(os.listdir(highres_split_dir))

            self.scenes = np.setdiff1d(self.scenes, high_res_list)
        offset = 0
        counts = []
        scenes = []
        sceneids = []
        images = []
        intrinsics = []
        trajectories = []
        groups = []
        id_ranges = []
        j = 0
        for scene_idx, scene in enumerate(self.scenes):
            scene_dir = osp.join(self.ROOT, split, scene)
            with np.load(
                osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True
            ) as data:
                imgs = data["images"]
                intrins = data["intrinsics"]
                traj = data["trajectories"]
                min_seq_len = self.min_views()
                if len(imgs) < min_seq_len:
                    print(f"Skipping {scene}")
                    continue

                collections = {}
                if "image_collection" not in data:
                    raise KeyError(
                        f"{scene}: 'image_collection' missing from "
                        "new_scene_metadata.npz"
                    )
                collections["image"] = data["image_collection"]

                num_imgs = imgs.shape[0]
                img_groups = []
                min_group_len = self.min_views()
                for ref_id, group in collections["image"].item().items():
                    if len(group) + 1 < min_group_len:
                        continue

                    # groups are (idx, score)s
                    group.insert(0, (ref_id, 1.0))
                    group = [int(x[0] + offset) for x in group]
                    img_groups.append(sorted(group))

                if len(img_groups) == 0:
                    print(f"Skipping {scene}")
                    continue

                scenes.append(scene)
                sceneids.extend([j] * num_imgs)
                id_ranges.extend([(offset, offset + num_imgs) for _ in range(num_imgs)])
                images.extend(imgs)
                intrinsics.extend(list(intrinsics_rows_to_K(intrins)))
                trajectories.extend(list(traj))

                # offset groups
                groups.extend(img_groups)
                counts.append(offset)
                offset += num_imgs
                j += 1

        if not scenes:
            raise EmptyDatasetError(
                f"ARKitScenes found no usable scenes under {osp.join(self.ROOT, split)}"
            )
        self.scenes = scenes
        self.sceneids = sceneids
        self.id_ranges = id_ranges
        self.images = images
        self.intrinsics = intrinsics
        self.trajectories = trajectories
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        # ARKitScenes lowres has TWO samplers: the temporal one (below) and the
        # pairs/`groups` one (the else branch), which draws a spatial collection
        # and permutes it -- it never calls get_seq_from_start_id, so the
        # stride/order policy does not reach it. Under self.sequential that
        # branch would silently hand back out-of-order frames and break the
        # consecutive-frame guarantee the TEST split is validated for, so take
        # the temporal branch unconditionally there (short-circuits before the
        # coin flip, so no rng draw is consumed either).
        if self.sequential or rng.choice([True, False]):
            image_idxs = np.arange(self.id_ranges[idx][0], self.id_ranges[idx][1])
            # nview from the batched sampler can be < self.num_views, so cut on
            # the ACTUAL clip length; min_views() is the scene-level floor
            start_image_idxs = image_idxs[: len(image_idxs) - num_views + 1]
            start_id = rng.choice(start_image_idxs)
            pos, ordered_video = self.get_seq_from_start_id(
                num_views,
                start_id,
                image_idxs.tolist(),
                rng,  # stride/order policy: self.stride_range + base defaults
            )
            image_idxs = np.array(image_idxs)[pos]
        else:
            # The `groups` sampler: a co-visible COLLECTION rather than a time
            # window -- wide baselines and irregular gaps, which is the point of
            # having two samplers here. The random draw picks WHICH frames; it
            # must not pick their order, so the result is sorted back into
            # capture order (groups are built sorted; permutation is what
            # destroyed it). Same invariant get_seq_from_start_id enforces:
            # a causal KV cache cannot be fed a clip that jumps back in time.
            # ordered_video stays False -- ordered, but not a video: the gaps
            # are arbitrary, so temporal losses/metrics must not treat it as one.
            ordered_video = False
            image_idxs = self.groups[idx]
            image_idxs = rng.permutation(image_idxs)
            if len(image_idxs) > num_views:
                image_idxs = image_idxs[:num_views]
            else:
                if rng.random() < 0.8:
                    image_idxs = rng.choice(image_idxs, size=num_views, replace=True)
                else:
                    repeat_num = num_views // len(image_idxs) + 1
                    image_idxs = np.tile(image_idxs, repeat_num)[:num_views]
            image_idxs = np.sort(image_idxs)

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            # frames live either in the scene dir (extracted layout) or in
            # its frames.zip (inode-safe layout); metadata npz reads above
            # are unaffected (always real files in the scene dir)
            scene_dir = frames_root(
                osp.join(self.ROOT, self.split_dir, self.scenes[scene_id])
            )

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]
            if basename[:8] != self.scenes[scene_id]:
                raise RuntimeError(
                    f"ARKitScenes frame/scene mismatch: basename {basename!r} "
                    f"does not belong to scene {self.scenes[scene_id]!r}"
                )
            # Load RGB image
            rgb_image = imread_cv2(
                osp.join(scene_dir, "vga_wide", basename.replace(".png", ".jpg"))
            )
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(scene_dir, "lowres_depth", basename), cv2.IMREAD_UNCHANGED
            )
            depthmap = depthmap.astype(np.float32) / 1000.0
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

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
                    dataset="arkitscenes",
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
                f"ARKitScenes produced {len(views)} views but {num_views} were requested"
            )
        return views
