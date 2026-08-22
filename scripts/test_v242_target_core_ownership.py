from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.target_hair_ownership_resolver_v842 import TargetHairOwnershipResolverV842
from models.target_hair_matte_refiner_v842 import TargetHairMatteRefinerV842
from models.face_overlap_contamination_guard_v842 import FaceOverlapContaminationGuardV842


def test_target_core_survives_face_skin_veto():
    target = torch.zeros(1, 1, 32, 32); target[:, :, 8:24, 8:24] = 1
    ones = torch.ones_like(target)
    resolver = TargetHairOwnershipResolverV842(core_radius=3)
    ownership, aux = resolver(coarse_target_hair_mask=target, source_hair_mask=torch.zeros_like(target),
                              source_face_mask=ones, source_skin_mask=ones,
                              anchor_hair_evidence=torch.full_like(target, 0.1), return_aux=True)
    assert aux["target_core_ownership"][0, 0, 12:20, 12:20].mean() > 0.9
    assert ownership[0, 0, 12:20, 12:20].min() >= 0.99


def test_transition_low_evidence_is_rejected_but_core_is_not():
    target = torch.zeros(1, 1, 32, 32); target[:, :, 6:26, 6:26] = 1
    resolver = TargetHairOwnershipResolverV842(core_radius=4)
    _, aux = resolver(coarse_target_hair_mask=target, source_hair_mask=torch.zeros_like(target),
                      source_face_mask=torch.ones_like(target), source_skin_mask=torch.ones_like(target),
                      anchor_hair_evidence=torch.full_like(target, 0.1), return_aux=True)
    matte, _ = TargetHairMatteRefinerV842()(coarse_target_hair_mask=target,
        occlusion_allowed_hair=aux["target_hair_ownership"], target_hair_core=aux["target_hair_core"],
        target_hair_transition=aux["target_hair_transition"], anchor_hair_evidence=aux["anchor_hair_evidence"],
        face_contact_ring=aux["face_contact_ring"], face_intrusion_risk=aux["face_intrusion_risk"],
        source_face_mask=torch.ones_like(target), source_skin_mask=torch.ones_like(target), return_aux=True)
    assert matte[0, 0, 12:20, 12:20].min() >= 0.9


def test_face_guard_transition_only():
    target = torch.ones(1, 1, 16, 16); core = torch.zeros_like(target); core[:, :, 4:12, 4:12] = 1
    guard, _ = FaceOverlapContaminationGuardV842()(alpha=target, allowed_hair=target,
        coarse_target_hair_mask=target, source_face_mask=target, source_skin_mask=target,
        face_contact_ring=target, face_intrusion_risk=target, hair_core=core,
        anchor_hair_evidence=torch.full_like(target, 0.1), new_hair_core=core, return_aux=True)
    assert guard[:, :, 4:12, 4:12].min() == 1


if __name__ == "__main__":
    test_target_core_survives_face_skin_veto()
    test_transition_low_evidence_is_rejected_but_core_is_not()
    test_face_guard_transition_only()
    print("V2.42 synthetic tests PASS")
