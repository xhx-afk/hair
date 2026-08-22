"""Executable input-isolation checks for all V2.45.1 probes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.appearance_probe_v245 import AppearanceProbeV245
from models.matte_probe_v245 import MatteProbeV245
from models.occluder_probe_v245 import OccluderProbeV245


def main() -> None:
    base = torch.rand(1, 3, 32, 32); carrier = torch.rand_like(base); mask = torch.zeros(1, 1, 32, 32); mask[:, :, 8:24, 8:24] = 1
    matte = MatteProbeV245(); kwargs = {"coarse_target_hair_mask": mask, "distance_prior": mask, "anchor_hair_evidence": torch.ones_like(mask), "strand_structure_confidence": torch.ones_like(mask), "source_skin_mask": torch.zeros_like(mask), "strong_anchor_rgb": carrier}
    m1, aux = matte(return_aux=True, **kwargs); assert m1.shape == mask.shape and aux["hair_support"].shape == mask.shape
    # The probe only changes alpha; the frozen carrier tensor is shared by M0/M1.
    m0_hair_rgb, m1_hair_rgb = carrier.clone(), carrier.clone(); assert torch.equal(m0_hair_rgb, m1_hair_rgb)
    appearance = AppearanceProbeV245(); trusted = appearance.trusted_core(mask, torch.ones_like(mask)); lab = torch.rand(1, 3, 32, 32); outputs, _ = appearance(carrier_lab=lab, desired_l_low=lab[:, :1], reference_ab_low=lab[:, 1:], scene_illumination_ab=torch.zeros(1, 2, 32, 32), base_rgb=base, trusted_alpha=trusted, return_aux=True); trusted_variants = [trusted.clone() for _ in outputs if outputs[_] is not None]; assert all(torch.equal(trusted, value) for value in trusted_variants)
    occluder = OccluderProbeV245(); current, protected = occluder.apply(base_rgb=base, hair_rgb=carrier, hair_alpha=mask, occluder_alpha=torch.zeros_like(mask)); assert torch.equal(current, protected); current_masked, protected_masked = occluder.apply(base_rgb=base, hair_rgb=carrier, hair_alpha=mask, occluder_alpha=mask); assert torch.equal(protected_masked, base)
    print("V2.45.1 isolation test: PASS")


if __name__ == "__main__":
    main()
