import torch

from .base import Details, MultiLoss, View
from .head_loss import CameraLoss, DepthOrPmapLoss, TrackLoss


class DistillLoss(MultiLoss):
    def __init__(self, lambda_track: float = 0.05) -> None:
        super().__init__()
        self.cam_loss = CameraLoss(delta=0.1, weights=(1.0, 1.0, 0.5))
        self.depth_loss = DepthOrPmapLoss(alpha=0.1)  # init 0.01 now 0.1
        self.pmap_loss = DepthOrPmapLoss(alpha=0.1)
        self.track_loss = TrackLoss()
        self.lambda_track = lambda_track

    def get_name(self) -> str:
        return "DistillLoss"

    def compute_loss(
        self,
        gts: list[View],
        preds: list[View],
        track_queries: torch.Tensor | None = None,
        track_preds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Details]:
        # ---------- Lcamera ----------
        cam_gt = torch.stack([g["camera_pose"] for g in gts], dim=1)
        cam_pr = torch.stack([p["camera_pose"] for p in preds], dim=1)
        Lcamera = self.cam_loss(cam_pr, cam_gt)

        # ---------- Ldepth ----------
        depth_terms = []
        for g, p in zip(gts, preds):
            if ("depth" in g) and ("depth" in p):
                sigma_p = p["depth_conf"]
                sigma_g = g["depth_conf"]
                valid_mask = g["valid_mask"]
                if not valid_mask.any():
                    valid_mask = torch.ones_like(g["valid_mask"])
                depth_terms.append(
                    self.depth_loss(
                        p["depth"], g["depth"], sigma_p, sigma_g, valid_mask
                    )
                )
        Ldepth = (
            torch.stack(depth_terms).mean()
            if depth_terms
            else torch.zeros_like(Lcamera)
        )

        # ---------- Lpmap ----------
        pmap_terms = []
        for g, p in zip(gts, preds):
            sigma_p = p["conf"]
            sigma_g = g["conf"]
            valid_mask = g["valid_mask"]
            if not valid_mask.any():
                valid_mask = torch.ones_like(g["valid_mask"])
            pmap_terms.append(
                self.pmap_loss(
                    p["pts3d_in_other_view"],
                    g["pts3d_in_other_view"],
                    sigma_p,
                    sigma_g,
                    valid_mask,
                )
            )
        Lpmap = torch.stack(pmap_terms).mean()

        # ---------- Ltrack ----------
        if ("track" in gts[0]) and ("track" in preds[0]):
            y_gt = torch.stack([g["track"] for g in gts], dim=1)
            vis_gt = torch.stack([g["vis"] for g in gts], dim=1)

            y_pr = torch.stack([p["track"] for p in preds], dim=1)
            vis_pr = torch.stack([p["vis"] for p in preds], dim=1)

            w_p = torch.stack([p["track_conf"] for p in preds], dim=1)
            w_g = torch.stack([g["track_conf"] for g in gts], dim=1)

            Ltrack = self.track_loss(y_pr, y_gt, vis_pr, vis_gt, w_p, w_g)
        else:
            Ltrack = torch.zeros_like(Lcamera)

        total = (
            Lcamera * 20 + Ldepth * 20 + Lpmap * 10 + self.lambda_track * 10 * Ltrack
        )
        details = {}

        details["Lcamera"] = float(Lcamera) * 20
        details["Ldepth"] = float(Ldepth) * 20
        details["Lpmap"] = float(Lpmap) * 10
        details["Ltrack"] = float(Ltrack) * self.lambda_track * 10
        details["total"] = float(total)

        return total, details
