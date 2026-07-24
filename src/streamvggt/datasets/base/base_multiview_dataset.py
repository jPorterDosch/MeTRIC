# Base multi-view dataset for the streamvggt dataset pipeline.
#
# Copied from DUSt3R (dust3r.datasets.base.base_multiview_dataset); the only
# changes are the import paths, now sourced from the streamvggt dataset package.
import itertools

import numpy as np
import PIL
import torch

from ..types import Split
from ..utils import cropping
from ..utils.corr import extract_correspondences_from_pts3d
from ..utils.geometry import depthmap_to_absolute_camera_coordinates
from ..utils.transforms import ImgNorm, SeqColorJitter
from .easy_dataset import EasyDataset


class EmptyDatasetError(RuntimeError):
    """A dataset root exists but yielded no usable scenes (stub or partially
    downloaded tree). Raised at construction so the misconfiguration surfaces
    immediately instead of as an IndexError at sampling time; catch by type,
    not by message text."""


def validate_stride_range(stride_range, dataset_name):
    """Shared guard for every stride_range entry point (dataset constructors
    AND DatasetConfig.validate): must be a (lo, hi) pair of integers with
    1 <= lo <= hi. Returns a plain tuple of builtin ints, so equality checks
    against (1, 1) hold regardless of the input container or integer type
    (a numpy int from a config sweep compares equal but is not `int`)."""
    try:
        lo, hi = stride_range
    except (TypeError, ValueError):
        raise ValueError(
            f"{dataset_name} stride_range must be a (lo, hi) pair, got {stride_range!r}"
        ) from None
    if (
        not isinstance(lo, (int, np.integer))
        or not isinstance(hi, (int, np.integer))
        or lo < 1
        or lo > hi
    ):
        raise ValueError(
            f"{dataset_name} stride_range must be ints with 1 <= lo <= hi, "
            f"got {stride_range!r}"
        )
    return (int(lo), int(hi))


def intrinsics_rows_to_K(intrins):
    """Build (N, 3, 3) pinhole K matrices from (N, 6) rows laid out as
    (w, h, fx, fy, cx, cy) -- the scene_metadata.npz convention shared by the
    ARKitScenes preprocess outputs."""
    intrins = np.asarray(intrins)
    K = np.expand_dims(np.eye(3), 0).repeat(len(intrins), 0)
    K[:, 0, 0] = intrins[:, 2]
    K[:, 1, 1] = intrins[:, 3]
    K[:, 0, 2] = intrins[:, 4]
    K[:, 1, 2] = intrins[:, 5]
    return K


def get_ray_map(c2w1, c2w2, intrinsics, h, w):
    c2w = np.linalg.inv(c2w1) @ c2w2
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    grid = np.stack([i, j, np.ones_like(i)], axis=-1)
    ro = c2w[:3, 3]
    rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
    rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
    rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.broadcast_to(ro, (h, w, 3))
    ray_map = np.concatenate([ro, rd], axis=-1)
    return ray_map


class BaseMultiViewDataset(EasyDataset):
    """Define all basic options.

    Usage:
        class MyDataset (BaseMultiViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(
        self,
        *,  # only keyword arguments
        num_views=None,
        split=None,
        resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
        stride_range=(1, 1),
        regular_stride=True,
        transform=ImgNorm,
        aug_crop=False,
        n_corres=0,
        nneg=0,
        seed=None,
        seq_aug_crop=False,
    ):
        assert num_views is not None, "undefined num_views"
        self.num_views = num_views
        self.split = split
        self._set_resolutions(resolution)

        self.n_corres = n_corres
        self.nneg = nneg
        assert (
            self.n_corres == "all"
            or isinstance(self.n_corres, int)
            or (
                isinstance(self.n_corres, list) and len(self.n_corres) == self.num_views
            )
        ), (
            f"Error, n_corres should either be 'all', a single integer or a list of length {self.num_views}"
        )
        assert self.nneg == 0 or self.n_corres != "all", (
            "nneg should be 0 if n_corres is all"
        )

        self.is_seq_color_jitter = False
        if isinstance(transform, str):
            raise TypeError(
                "transform must be a callable (e.g. ImgNorm/ColorJitter) or SeqColorJitter; string transforms are not supported"
            )
        if transform == SeqColorJitter:
            transform = SeqColorJitter()
            self.is_seq_color_jitter = True
        self.transform = transform

        self.aug_crop = aug_crop
        self.seed = seed
        self.seq_aug_crop = seq_aug_crop
        # Per-clip frame stride (lo, hi): one stride is drawn uniformly in
        # [lo, hi] per clip (see get_seq_from_start_id). (1, 1) is exactly
        # consecutive frames. TEST is pinned to (1, 1): under a stochastic
        # stride, "adjacent" clip entries can be 20 source frames apart, which
        # silently turns TAE/frame-scrubbing into cross-view readings.
        self.stride_range = validate_stride_range(stride_range, type(self).__name__)
        if self.split == Split.TEST and self.stride_range != (1, 1):
            raise ValueError(
                f"{type(self).__name__}: TEST split requires stride_range=(1, 1) "
                f"-- temporal metrics (TAE, warp consistency) assume consecutive "
                f"frames; got {self.stride_range}"
            )
        # True: ONE stride per clip == a constant frame rate. False: gaps drawn
        # per adjacent pair, so the rate varies within the clip. A deliberate
        # config choice, not a per-sample coin flip -- it is an ablation axis.
        if not isinstance(regular_stride, bool):
            raise ValueError(
                f"{type(self).__name__} regular_stride must be a bool, "
                f"got {regular_stride!r}"
            )
        self.regular_stride = regular_stride
        # plain attribute (not a property) so diagnostics like
        # tests/temp_mask_survival.py can flip it for A/B runs
        self.sequential = self.stride_range == (1, 1)

    def min_views(self):
        """Minimum frames a scene needs to be usable. Every clip is drawn from
        distinct frames, so this is exactly num_views: shorter scenes are
        skipped at load time rather than padded by repetition."""
        return self.num_views

    def __len__(self):
        return len(self.scenes)

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        stride_range=None,
    ):
        """Sample num_views positions starting from id_ref.

        TEMPORAL ORDER IS AN INVARIANT, not a sampling mode: every return path
        is sorted ascending. The model is causal and reads a KV cache built
        frame by frame, so an out-of-order clip trains it on a history that
        cannot occur at deployment. (The dust3r/VGGT original made this a
        coin flip -- video_prob / block_shuffle -- which is why those knobs and
        blockwise_shuffle are gone: they existed only to emit shuffled clips.)

        What varies is the STRIDE, i.e. how far apart the ordered frames sit.
        Two config-selected regimes (self.regular_stride, never a coin flip --
        it is an ablation axis, so a run must be one or the other):
          regular (default) -> ONE stride for the whole clip, drawn uniformly
            from stride_range: a constant frame rate. Keeps motion-speed
            augmentation and baseline/parallax diversity while staying
            deployment-ordered.
          irregular -> gaps drawn independently per adjacent pair, so the frame
            rate varies WITHIN the clip.
        Keep hi small (~3-5) for near-video; (1, 1) = pure consecutive (the
        TEST policy, enforced at construction).

        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            rng: random number generator
            stride_range: (lo, hi) frame-stride bounds; None -> the dataset's
                own self.stride_range. Scenes too short for lo fall back to
                the largest feasible stride.
        returns:
            pos: list of positions of the views in ids_all, i.e., index for
                ids_all -- always ascending
            is_video: kept for the view-dict field of the same name; always
                True now that ordering is guaranteed
        """
        if stride_range is None:
            stride_range = self.stride_range
        elif self.sequential and validate_stride_range(
            stride_range, type(self).__name__
        ) != (1, 1):
            # The consecutive-mode early return below keys off self.sequential,
            # so honoring it would silently discard this override and hand back
            # stride-1 clips. Fail loudly: an ablation arm that thinks it is
            # sampling at stride 20 but is not can waste a whole training run.
            raise ValueError(
                f"{type(self).__name__}: stride_range={tuple(stride_range)} was "
                f"passed to get_seq_from_start_id but the dataset is in "
                f"consecutive mode (self.sequential, from stride_range="
                f"{self.stride_range}); the override would be ignored. Rebuild "
                f"the dataset with the stride you want, or clear self.sequential."
            )
        min_interval, max_interval = validate_stride_range(
            stride_range, type(self).__name__
        )
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)

        # Consecutive mode (stride pinned to exactly 1): temporal order
        # preserved, no rng draws. The stochastic sampling below is multi-view
        # augmentation inherited from the dust3r/VGGT recipe -- right for
        # learning parallax, but it breaks anything assuming pixel-aligned
        # temporal adjacency: the TGM-style temporal loss, TAE, and the
        # follow-camera visualization. See self.sequential.
        if self.sequential and len(ids_all) >= num_views:
            # slide the window back if it would run off the end of the scene
            start = min(pos_ref, len(ids_all) - num_views)
            return [start + i for i in range(num_views)], True

        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            # cap the stride so the clip fits in the scene, then drop the
            # floor to match when the scene is too short for the requested
            # minimum (feasibility beats the configured lower bound)
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            lo = min(min_interval, max_interval)

            # `intervals` holds the GAPS between consecutive picks; accumulate()
            # turns them into absolute positions below. Build only the one the
            # configured mode uses -- the other is pure waste (and would still
            # burn rng draws).
            if self.regular_stride:
                # regular: ONE stride for the whole clip == a constant frame
                # rate. Capped so num_views-1 steps still fit inside the scene,
                # hence no overshoot and no backfill below.
                hi = min(remaining_sum // (num_views - 1), max_interval)
                fixed_interval = rng.choice(range(min(lo, hi), hi + 1))
                intervals = [fixed_interval for _ in range(num_views - 1)]
            else:
                # irregular: each gap drawn independently, so the frame rate
                # varies within the clip. max_interval carries a factor 2 above
                # (gaps average out rather than each fitting), so the walk CAN
                # run past the end of the scene -- the filter/backfill below
                # exists for exactly this case.
                intervals = [
                    rng.choice(range(lo, max_interval + 1))
                    for _ in range(num_views - 1)
                ]

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = (
                pos
                + rng.choice(
                    pos_candidates, num_views - len(pos), replace=False
                ).tolist()
            )

            # sorted unconditionally: the backfill above appends arbitrary
            # positions, and order is an invariant (see docstring)
            pos = sorted(pos)
            is_video = True
        else:
            # Unreachable by construction: every loader builds its start ids as
            # img_ids[: num_imgs - num_views + 1] and skips scenes shorter than
            # min_views(), and __getitem__ asserts nview <= self.num_views, so
            # id_ref always leaves at least num_views-1 frames behind it. This
            # used to be the allow_repeat path, which padded a short scene by
            # REPEATING frames -- dropped along with the flag: a clip with
            # duplicate frames is not something the streaming model ever sees.
            raise ValueError(
                f"{type(self).__name__}: only {remaining_sum + 1} frames from "
                f"id_ref={id_ref} but {num_views} views requested. The loader "
                f"should not have offered this start id -- check its start-id "
                f"construction and min_views() skip."
            )
        assert len(pos) == num_views
        return pos, is_video

    def get_img_and_ray_masks(self, is_metric, v, rng, p=[0.8, 0.15, 0.05]):
        # generate img mask and raymap mask
        if v == 0 or (not is_metric):
            img_mask = True
            raymap_mask = False
        else:
            rand_val = rng.random()
            if rand_val < p[0]:
                img_mask = True
                raymap_mask = False
            elif rand_val < p[0] + p[1]:
                img_mask = False
                raymap_mask = True
            else:
                img_mask = True
                raymap_mask = True
        return img_mask, raymap_mask

    def get_stats(self):
        return f"{len(self)} groups of views"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace("self.", "")
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, resolution, rng, num_views):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list, np.ndarray)):
            # the idx is specifying the aspect-ratio
            idx, ar_idx, nview = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            nview = self.num_views

        assert nview >= 1 and nview <= self.num_views
        # set-up the rng
        # `is not None`, not truthiness: seed=0 is a legitimate seed, and the
        # falsy check silently demoted it to the unseeded branch below
        if self.seed is not None:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        if self.aug_crop > 1 and self.seq_aug_crop:
            self.delta_target_resolution = self._rng.integers(0, self.aug_crop)

        # over-loaded code
        resolution = self._resolutions[
            ar_idx
        ]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng, nview)
        assert len(views) == nview

        if "camera_pose" not in views[0]:
            views[0]["camera_pose"] = np.ones((4, 4), dtype=np.float32)
        first_view_camera_pose = views[0]["camera_pose"]
        transform = SeqColorJitter() if self.is_seq_color_jitter else self.transform

        for v, view in enumerate(views):
            assert "pts3d" not in view, (
                f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            )
            view["idx"] = (idx, ar_idx, v)

            # encode the image
            width, height = view["img"].size

            view["true_shape"] = np.int32((height, width))
            view["img"] = transform(view["img"])
            view["sky_mask"] = view["depthmap"] < 0

            assert "camera_intrinsics" in view
            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view["camera_pose"]).all(), (
                    f"NaN in camera pose for view {view_name(view)}"
                )

            ray_map = get_ray_map(
                first_view_camera_pose,
                view["camera_pose"],
                view["camera_intrinsics"],
                height,
                width,
            )
            view["ray_map"] = ray_map.astype(np.float32)

            assert "pts3d" not in view
            assert "valid_mask" not in view
            assert np.isfinite(view["depthmap"]).all(), (
                f"NaN in depthmap for view {view_name(view)}"
            )
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view["camera_intrinsics"]  # noqa: F841

        if self.n_corres > 0:
            ref_view = views[0]
            for view in views:
                corres1, corres2, valid = extract_correspondences_from_pts3d(
                    ref_view, view, self.n_corres, self._rng, nneg=self.nneg
                )
                view["corres"] = (corres1, corres2)
                view["valid_corres"] = valid

        # last thing done!
        for view in views:
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")
        return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), (
                f"Bad type for {width=} {type(width)=}, should be int"
            )
            assert isinstance(height, int), (
                f"Bad type for {height=} {type(height)=}, should be int"
            )
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        left, top = cx - min_margin_x, cy - min_margin_y
        right, bottom = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (left, top, right, bottom)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        # transpose the resolution if necessary
        W, H = image.size  # new size

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        return image, depthmap, intrinsics2


def is_good_type(key, v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view["true_shape"]

    if width < height:
        # rectify portrait to landscape
        assert view["img"].shape == (3, height, width)
        view["img"] = view["img"].swapaxes(1, 2)

        assert view["valid_mask"].shape == (height, width)
        view["valid_mask"] = view["valid_mask"].swapaxes(0, 1)

        assert view["depthmap"].shape == (height, width)
        view["depthmap"] = view["depthmap"].swapaxes(0, 1)

        assert view["pts3d"].shape == (height, width, 3)
        view["pts3d"] = view["pts3d"].swapaxes(0, 1)

        # transpose x and y pixels
        view["camera_intrinsics"] = view["camera_intrinsics"][[1, 0, 2]]

        assert view["ray_map"].shape == (height, width, 6)
        view["ray_map"] = view["ray_map"].swapaxes(0, 1)

        assert view["sky_mask"].shape == (height, width)
        view["sky_mask"] = view["sky_mask"].swapaxes(0, 1)

        if "corres" in view:
            # transpose correspondences x and y
            view["corres"][0] = view["corres"][0][:, [1, 0]]
            view["corres"][1] = view["corres"][1][:, [1, 0]]
