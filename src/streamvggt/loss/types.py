"""Typed, explicit configuration for training objectives.

The losses in this repo were refactored into individual files for readability;
this module is the single source of truth for *which* objectives exist and
*how* they are assembled. Historically the criterion was a Python snippet
(e.g. ``"ConfLoss(Regr3DPose(L21, norm_mode='?avg_dis'), alpha=0.2) +
FinetuneLoss()"``) that the entrypoints ``eval()``'d after a wildcard import.
That is replaced here by:

* :class:`LossName` -- the enumerated set of composable (``MultiLoss``) losses,
  so a loss can be referenced without ``eval`` over a wildcard import.
* :class:`PixelLoss` -- the per-pixel base distances usable as a regression arg.
* :class:`Recipe` -- the named, composed criteria the repo actually trains with.
* :class:`LossConfig` -- a dataclass exposing every tunable knob plus the recipe
  selector; :meth:`LossConfig.build` returns the assembled ``nn.Module``.

Enum ``.value``s are plain builtins (str) so ``train_utils.to_primitive`` can
serialize a config into the experiment manifest / checkpoint without needing
this package on the path to unpickle it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .base import BaseCriterion, MultiLoss
from .conf_loss import ConfLoss
from .depth_train_loss import DepthTrainLoss
from .distill_loss import DistillLoss
from .finetune_loss import FinetuneLoss
from .l_loss import L21, MSE
from .regr_3d_pose import Regr3DPose, Regr3DPose_ScaleInv, Regr3DPoseBatchList
from .rgb_loss import RGBLoss


class PixelLoss(str, Enum):
    """Per-pixel base distance passed into a regression loss (``Regr3DPose``,
    ``RGBLoss``). The value is the historical eval-token name."""

    L21 = "L21"
    MSE = "MSE"

    def build(self) -> BaseCriterion:
        return {PixelLoss.L21: L21, PixelLoss.MSE: MSE}[self]


class LossName(str, Enum):
    """Every eval-able (composable ``MultiLoss``) loss in the package.

    The value is the class name -- i.e. the exact token the old criterion
    strings referenced -- so names round-trip to/from the legacy format. Use
    :attr:`cls` to get the concrete class for construction.
    """

    RGB = "RGBLoss"
    REGR_3D_POSE = "Regr3DPose"
    REGR_3D_POSE_BATCH_LIST = "Regr3DPoseBatchList"
    REGR_3D_POSE_SCALE_INV = "Regr3DPose_ScaleInv"
    CONF = "ConfLoss"
    FINETUNE = "FinetuneLoss"
    DISTILL = "DistillLoss"

    @property
    def cls(self) -> type[MultiLoss]:
        return _LOSS_CLASSES[self]


_LOSS_CLASSES: dict[LossName, type[MultiLoss]] = {
    LossName.RGB: RGBLoss,
    LossName.REGR_3D_POSE: Regr3DPose,
    LossName.REGR_3D_POSE_BATCH_LIST: Regr3DPoseBatchList,
    LossName.REGR_3D_POSE_SCALE_INV: Regr3DPose_ScaleInv,
    LossName.CONF: ConfLoss,
    LossName.FINETUNE: FinetuneLoss,
    LossName.DISTILL: DistillLoss,
}


class Recipe(str, Enum):
    """The named, composed criteria the repo trains/evaluates with. Each maps to
    a fixed composition of :class:`LossName` losses; :class:`LossConfig` knobs
    parameterize the leaves."""

    DEPTH_TRAIN = "depth_train"
    DISTILL = "distill"
    FINETUNE_TRAIN = "finetune_train"
    FINETUNE_TEST = "finetune_test"


@dataclass
class LossConfig:
    """Explicit, serializable replacement for the eval'd criterion string.

    A :class:`Recipe` selects the composition; the remaining fields are the
    tunable constructor knobs threaded into the leaf losses by :meth:`build`.
    The plain-default config reproduces ``DEPTH_TRAIN`` (the default training
    objective); other recipes are built by passing ``recipe=`` plus whatever
    knobs they need, e.g. ``LossConfig(recipe=Recipe.FINETUNE_TRAIN)``. Recipes
    bind their own composition-intrinsic settings in :meth:`build` (e.g.
    FINETUNE_TEST forces ``gt_scale=True``/``sky_loss_value=0``), so they are
    correct when constructed bare.
    """

    recipe: Recipe = Recipe.DEPTH_TRAIN

    # Base pixel distance for the Regr3DPose regression terms.
    pixel_loss: PixelLoss = PixelLoss.L21

    # DepthTrainLoss knobs
    depth_weights: tuple[float, float] | None = None
    # NOTE: depth_alpha (the confidence-regularization weight) is scale-sensitive
    # and interacts with depth_metric. In scale-invariant mode residuals are
    # ~unit-normalized; in metric mode they are raw metres (much larger), which
    # swamps the -alpha*log(sigma) term -- retune depth_alpha (and depth_weights)
    # when flipping depth_metric.
    depth_alpha: float = 0.1
    depth_trim: float = 0.2
    temp_grad_scales: int = 4
    temp_grad_decay: float = 0.5
    reduction: str = "batch-based"
    diff_depth_th: float = 0.05
    # Supervise absolute (metric) depth: skip the scale/shift alignment in both
    # the depth and temporal terms. On by default because DEPTH_TRAIN targets
    # metric-depth-conditioned training; only DEPTH_TRAIN consumes this field.
    depth_metric: bool = True
    # Run the depth accuracy term on log-depth (|log pred - log gt|, relative &
    # scale-aware) instead of raw metres, so the far background stops dominating
    # the L1. Keeps the metric scale penalty (unlike scale-invariant SILog).
    # Retune depth_alpha when enabling: residuals shrink to ~relative units, so
    # the confidence scale (conf ~ alpha/err) shifts. Only DEPTH_TRAIN uses it.
    depth_log_space: bool = False

    # Regr3DPose / Regr3DPose_ScaleInv knobs.
    norm_mode: str = "?avg_dis"
    gt_scale: bool = False
    sky_loss_value: float = 2.0
    # 0.0 disables metric-scale capping (constructor treats falsy as off).
    max_metric_scale: float = 0.0

    # ConfLoss confidence-regularization weight.
    conf_alpha: float = 0.2

    # FinetuneLoss / DistillLoss track-loss weight.
    lambda_track: float = 0.05

    def __post_init__(self) -> None:
        # Accept plain strings for the enum fields so a config coming from
        # hydra/OmegaConf, JSON, or a checkpoint reconstructs without ceremony.
        # tyro already supplies enum members, in which case these are no-ops.
        self.recipe = Recipe(self.recipe)
        self.pixel_loss = PixelLoss(self.pixel_loss)

    def _regr(
        self,
        name: LossName,
        *,
        gt_scale: bool | None = None,
        sky_loss_value: float | None = None,
    ) -> Regr3DPose:
        """Build one Regr3DPose-family term with the shared regression knobs.

        ``gt_scale`` / ``sky_loss_value`` default to the config fields but can be
        overridden per term so a recipe can bind its composition-intrinsic
        settings (e.g. the eval metric pins gt_scale/sky regardless of config).
        """
        return name.cls(
            self.pixel_loss.build(),
            norm_mode=self.norm_mode,
            gt_scale=self.gt_scale if gt_scale is None else gt_scale,
            sky_loss_value=self.sky_loss_value
            if sky_loss_value is None
            else sky_loss_value,
            max_metric_scale=self.max_metric_scale or False,
        )

    def build(self) -> MultiLoss:
        """Assemble the criterion ``nn.Module`` described by this config."""
        match self.recipe:
            case Recipe.DEPTH_TRAIN:
                return DepthTrainLoss(
                    weights=self.depth_weights,
                    alpha=self.depth_alpha,
                    trim=self.depth_trim,
                    temp_grad_scales=self.temp_grad_scales,
                    temp_grad_decay=self.temp_grad_decay,
                    reduction=self.reduction,
                    diff_depth_th=self.diff_depth_th,
                    metric=self.depth_metric,
                    log_space=self.depth_log_space,
                )

            case Recipe.DISTILL:
                return DistillLoss(lambda_track=self.lambda_track)

            case Recipe.FINETUNE_TRAIN:
                regr = self._regr(LossName.REGR_3D_POSE)
                return ConfLoss(regr, alpha=self.conf_alpha) + FinetuneLoss(
                    lambda_track=self.lambda_track
                )

            case Recipe.FINETUNE_TEST:
                # Eval metric: use GT scale and drop the sky term. These are
                # intrinsic to the test protocol, so bind them here rather than
                # relying on the caller/yaml to pass the right knobs.
                test_kw = dict(gt_scale=True, sky_loss_value=0.0)
                return self._regr(LossName.REGR_3D_POSE, **test_kw) + self._regr(
                    LossName.REGR_3D_POSE_SCALE_INV, **test_kw
                )

            case _:
                raise ValueError(f"unknown recipe {self.recipe!r}")

    def describe(self) -> str:
        """The assembled criterion's legacy-style string (for logging)."""
        return repr(self.build())
