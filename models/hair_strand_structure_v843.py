"""V2.43 local strand evidence from the strong-anchor luminance field.

This is deliberately a small, deterministic image operator.  It is used as a
confidence signal, never as a hard mask, so weak or ambiguous hair remains
eligible for the transition path without becoming opaque by construction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _blur(value: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 0)
    if radius == 0:
        return value
    return F.avg_pool2d(value, 2 * radius + 1, stride=1, padding=radius)


def _sobel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype, device = value.dtype, value.device
    kx = value.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3)
    ky = value.new_tensor(((-1, -2, -1), (0, 0, 0), (1, 2, 1))).view(1, 1, 3, 3)
    del dtype, device
    return F.conv2d(F.pad(value, (1, 1, 1, 1), mode="replicate"), kx), F.conv2d(
        F.pad(value, (1, 1, 1, 1), mode="replicate"), ky
    )


def strand_structure_confidence_v843(
    anchor_l: torch.Tensor,
    *,
    local_hair_support: torch.Tensor | None = None,
    anchor_hair_evidence: torch.Tensor | None = None,
    blur_radius: int = 2,
) -> dict[str, torch.Tensor]:
    """Return structure-tensor anisotropy and a hair-weighted confidence map."""
    if anchor_l.dim() == 3:
        # [B,H,W] -> [B,1,H,W]
        anchor_l = anchor_l.unsqueeze(1)
    if anchor_l.dim() != 4:
        raise ValueError("anchor_l must have shape [B,1,H,W] or [B,H,W]")
    l = anchor_l[:, :1].float()
    gx, gy = _sobel(l)
    jxx, jyy, jxy = _blur(gx * gx, blur_radius), _blur(gy * gy, blur_radius), _blur(gx * gy, blur_radius)
    trace = jxx + jyy
    discr = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + 1e-8)
    lambda1 = 0.5 * (trace + discr)
    lambda2 = 0.5 * (trace - discr).clamp_min(0.0)
    anisotropy = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)).clamp(0, 1)
    support = torch.ones_like(anisotropy) if local_hair_support is None else local_hair_support.float().clamp(0, 1)
    evidence = torch.ones_like(anisotropy) if anchor_hair_evidence is None else anchor_hair_evidence.float().clamp(0, 1)
    if support.shape[-2:] != l.shape[-2:]:
        support = F.interpolate(support, size=l.shape[-2:], mode="bilinear", align_corners=False)
    if evidence.shape[-2:] != l.shape[-2:]:
        evidence = F.interpolate(evidence, size=l.shape[-2:], mode="bilinear", align_corners=False)
    confidence = (anisotropy * support * evidence).clamp(0, 1)
    return {
        "anchor_l": l,
        "gradient_x": gx,
        "gradient_y": gy,
        "structure_lambda1": lambda1,
        "structure_lambda2": lambda2,
        "anisotropy": anisotropy,
        "local_hair_support": support,
        "strand_structure_confidence": confidence,
    }


class HairStrandStructureV843:
    def __init__(self, *, blur_radius: int = 2) -> None:
        self.blur_radius = int(blur_radius)

    def __call__(self, anchor_l: torch.Tensor, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        return strand_structure_confidence_v843(anchor_l, blur_radius=self.blur_radius, **kwargs)


__all__ = ["HairStrandStructureV843", "strand_structure_confidence_v843"]
