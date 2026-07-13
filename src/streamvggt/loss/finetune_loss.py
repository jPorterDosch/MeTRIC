import torch
from .base import Details, MultiLoss, View
from .head_loss import CameraLoss, DepthOrPmapLoss
from streamvggt.utils.pose_enc import extri_intri_to_pose_encoding


class FinetuneLoss(MultiLoss):
    def __init__(self, lambda_track: float = 0.05) -> None:
        super().__init__()
        self.cam_loss = CameraLoss(delta=0.1, weights=(1.0, 1.0, 0.5))
        self.depth_loss = DepthOrPmapLoss(alpha=0.1)

    def get_name(self) -> str:
        return "FinetuneLoss"

    def compute_loss(
        self,
        gts: list[View],
        preds: list[View],
        track_queries: torch.Tensor | None = None,
        track_preds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Details]:
        # ---------- Lcamera ----------
        T = []
        for g in gts:
            T_c2w = g["camera_pose"]
            if not torch.is_tensor(T_c2w):
                T_c2w = torch.as_tensor(T_c2w)
            dtype = T_c2w.dtype
            device = T_c2w.device

            R = T_c2w[..., :3, :3]  # [...,3,3]
            t = T_c2w[..., :3, 3:4]  # [...,3,1]

            # c2w -> w2c: R^T, -R^T t
            R_w2c = R.transpose(-1, -2)  # [...,3,3]
            t_w2c = -(R_w2c @ t)  # [...,3,1]

            eye = torch.eye(4, dtype=dtype, device=device)
            T_w2c = eye.expand(*T_c2w.shape[:-2], 4, 4).clone()  # [...,4,4]
            T_w2c[..., :3, :3] = R_w2c
            T_w2c[..., :3, 3:4] = t_w2c

            if T_w2c.dim() == 2:
                T_w2c = T_w2c.unsqueeze(0)

            T.append(T_w2c)  # [B,4,4]

        T_view = torch.stack(T, dim=1)
        T_c2w_first = torch.inverse(T_view[:, 0])

        T_wprime2c = T_view @ T_c2w_first.unsqueeze(1)  # [B,V,4,4]
        camera_extrinsics_gt = T_wprime2c
        camera_intrinsics_gt = torch.stack(
            [g["camera_intrinsics"] for g in gts], dim=1
        )  # b v 3 3
        images_hw = gts[0]["img"].shape[-2:]
        cam_gt = extri_intri_to_pose_encoding(
            camera_extrinsics_gt, camera_intrinsics_gt, images_hw
        )
        cam_pr = torch.stack([p["camera_pose"] for p in preds], dim=1)

        Lcamera = self.cam_loss(cam_pr, cam_gt)

        # ---------- Ldepth ----------
        depth_terms = []
        for g, p in zip(gts, preds):
            if "depth" in p:
                sigma_p = p["depth_conf"]
                valid_mask = g["valid_mask"]
                if not valid_mask.any():
                    valid_mask = torch.ones_like(g["valid_mask"])
                depth_terms.append(
                    self.depth_loss(
                        p["depth"],
                        g["depthmap"].unsqueeze(-1),
                        sigma_p=sigma_p,
                        valid_mask=valid_mask,
                    )
                )
        Ldepth = (
            torch.stack(depth_terms).mean()
            if depth_terms
            else torch.zeros_like(Lcamera)
        )

        total = Lcamera * 20 + Ldepth * 10
        details = {}

        details["Lcamera"] = float(Lcamera) * 20
        details["Ldepth"] = float(Ldepth) * 10
        details["total"] = float(total)

        return total, details
