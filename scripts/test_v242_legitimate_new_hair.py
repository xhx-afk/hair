import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.target_hair_ownership_resolver_v842 import TargetHairOwnershipResolverV842


def test_legitimate_new_hair_core():
    target = torch.ones(1, 1, 24, 24)
    _, aux = TargetHairOwnershipResolverV842(core_radius=3)(
        coarse_target_hair_mask=target, source_hair_mask=torch.zeros_like(target),
        source_face_mask=torch.ones_like(target), source_skin_mask=torch.ones_like(target),
        anchor_hair_evidence=torch.full_like(target, .1), return_aux=True)
    assert aux["new_hair_core"][:, :, 8:16, 8:16].min() == 1


if __name__ == "__main__":
    test_legitimate_new_hair_core()
    print("V2.42 legitimate new hair PASS")
