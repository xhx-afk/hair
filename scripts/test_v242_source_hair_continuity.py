import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.target_hair_ownership_resolver_v842 import TargetHairOwnershipResolverV842


def test_source_hair_continuity():
    target = torch.ones(1, 1, 20, 20)
    source = torch.zeros_like(target); source[:, :, 8:12, :] = 1
    own, _ = TargetHairOwnershipResolverV842(core_radius=2)(
        coarse_target_hair_mask=target, source_hair_mask=source,
        source_face_mask=torch.zeros_like(target), source_skin_mask=torch.zeros_like(target),
        anchor_hair_evidence=torch.ones_like(target), return_aux=True)
    assert torch.isfinite(own).all()


if __name__ == "__main__":
    test_source_hair_continuity()
    print("V2.42 source hair continuity PASS")
