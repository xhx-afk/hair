"""V2.41 reference palette with one stable hue direction."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab


class ReferenceStableHuePaletteV841:
    def __init__(self, *, chroma_valid_threshold: float = 3.0,
                 mad_scale: float = 3.5, min_support: int = 16) -> None:
        self.chroma_valid_threshold = float(chroma_valid_threshold)
        self.mad_scale = float(mad_scale)
        self.min_support = int(min_support)

    @staticmethod
    def _quantiles(value: torch.Tensor, q: list[float]) -> torch.Tensor:
        return torch.quantile(value, value.new_tensor(q))

    def __call__(self, *, reference_rgb: torch.Tensor,
                 reference_hair_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        lab = rgb_to_lab(reference_rgb)
        mask = reference_hair_mask.float().clamp(0, 1) > 0.5
        stable_dirs, reliabilities, dispersions, chromas = [], [], [], []
        valid_fractions, stable_supports = [], []
        l_quantiles, c_groups = [], [[], [], []]
        for batch in range(lab.size(0)):
            pixels = lab[batch].permute(1, 2, 0)[mask[batch, 0]]
            if pixels.numel() == 0:
                pixels = lab.new_zeros((1, 3))
            ab = pixels[:, 1:]
            c = torch.linalg.vector_norm(ab, dim=1)
            c_med = c.median()
            c_mad = (c - c_med).abs().median().clamp_min(1.0)
            robust = (c - c_med).abs() <= self.mad_scale * c_mad
            if not robust.any():
                robust = torch.ones_like(c, dtype=torch.bool)
            valid = robust & (c >= self.chroma_valid_threshold)
            hue_keep_full = robust.clone()
            if not valid.any():
                stable = ab.new_zeros(2)
                dispersion = ab.new_tensor(0.0)
                filtered_c = c[robust]
                reliability = ab.new_tensor(0.0)
            else:
                valid_ab, valid_c = ab[valid], c[valid]
                unit = valid_ab / valid_c[:, None].clamp_min(1e-5)
                weights = valid_c.clamp(0.0, 30.0)
                initial = (unit * weights[:, None]).sum(0)
                initial = initial / torch.linalg.vector_norm(initial).clamp_min(1e-5)
                cosine = (unit * initial).sum(1).clamp(-1.0, 1.0)
                angles = torch.rad2deg(torch.acos(cosine))
                med = angles.median()
                mad = (angles - med).abs().median().clamp_min(1.0)
                keep = angles <= med + self.mad_scale * mad
                if not keep.any():
                    keep = torch.ones_like(angles, dtype=torch.bool)
                valid_indices = torch.where(valid)[0]
                hue_keep_full[valid_indices] = keep
                unit, valid_c, weights, angles = unit[keep], valid_c[keep], weights[keep], angles[keep]
                stable = (unit * weights[:, None]).sum(0)
                stable = stable / torch.linalg.vector_norm(stable).clamp_min(1e-5)
                cosine = (unit * stable).sum(1).clamp(-1.0, 1.0)
                dispersion = torch.rad2deg(torch.acos(cosine)).mul(weights).sum() / weights.sum().clamp_min(1e-5)
                filtered_c = c[robust]
                support = min(float(valid_c.numel()) / max(self.min_support, 1), 1.0)
                reliability = (ab.new_tensor(support) * keep.float().mean()).clamp(0, 1)
            l = pixels[:, 0]
            q = self._quantiles(l, [0.10, 0.25, 0.50, 0.75, 0.90])
            l_quantiles.append(q)
            chromas.append(filtered_c.median() if filtered_c.numel() else c.median())
            bounds = self._quantiles(l, [0.25, 0.75])
            groups = (l <= bounds[0]), ((l > bounds[0]) & (l < bounds[1])), (l >= bounds[1])
            for group, bucket in zip(groups, c_groups):
                values = c[group & hue_keep_full]
                bucket.append(values.median() if values.numel() else c.median())
            stable_dirs.append(stable)
            reliabilities.append(reliability)
            dispersions.append(dispersion)
            valid_fractions.append(valid.float().mean())
            stable_supports.append(valid.float().sum())

        q = torch.stack(l_quantiles).view(-1, 5, 1, 1)
        stable = torch.stack(stable_dirs).view(-1, 2, 1, 1)
        global_chroma = torch.stack(chromas).view(-1, 1, 1, 1)
        result = {
            "stable_hue_unit_ab": stable,
            "global_chroma": global_chroma,
            "shadow_chroma": torch.stack(c_groups[0]).view(-1, 1, 1, 1),
            "mid_chroma": torch.stack(c_groups[1]).view(-1, 1, 1, 1),
            "highlight_chroma": torch.stack(c_groups[2]).view(-1, 1, 1, 1),
            "l_q10": q[:, 0:1], "l_q25": q[:, 1:2], "l_q50": q[:, 2:3],
            "l_q75": q[:, 3:4], "l_q90": q[:, 4:5],
            "palette_reliability": torch.stack(reliabilities).view(-1, 1, 1, 1),
            "hue_dispersion_deg": torch.stack(dispersions).view(-1, 1, 1, 1),
            "valid_hue_fraction": torch.stack(valid_fractions).view(-1, 1, 1, 1),
            "stable_hue_support_count": torch.stack(stable_supports).view(-1, 1, 1, 1),
            "reference_hair_count": mask.flatten(1).sum(1, keepdim=True).view(-1, 1, 1, 1),
            "reference_lab": lab,
        }
        result["hue_metric_valid"] = (
            (result["stable_hue_support_count"] >= float(self.min_support))
            & (result["palette_reliability"] >= 0.35)
        ).float()
        preview_l = result["l_q50"]
        preview_ab = stable * global_chroma
        result["stable_hue_preview_rgb"] = lab_to_rgb(torch.cat((preview_l, preview_ab), dim=1)).clamp(0, 1)
        result["stable_hue_stability"] = (1.0 - result["hue_dispersion_deg"] / 45.0).clamp(0, 1)
        return result


__all__ = ["ReferenceStableHuePaletteV841"]
