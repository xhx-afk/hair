"""V2.42 target-hair core ownership and legitimate new-growth path."""

from __future__ import annotations

from models.face_overlap_contamination_guard_v842 import FaceOverlapContaminationGuardV842
from models.hair_local_recomposition_v841 import HairLocalRecompositionV841
from models.target_hair_matte_refiner_v842 import TargetHairMatteRefinerV842
from models.target_hair_ownership_resolver_v842 import TargetHairOwnershipResolverV842


class HairLocalRecompositionV842(HairLocalRecompositionV841):
    def __init__(self, **kwargs) -> None:
        kwargs["version"] = "v2.42"
        super().__init__(**kwargs)
        self.resolver = TargetHairOwnershipResolverV842(
            contact_radius=kwargs.get("contact_radius", 5),
            core_radius=kwargs.get("matte_ring_radius", 5),
        )
        self.matte = TargetHairMatteRefinerV842(core_floor=0.90)
        self.guard = FaceOverlapContaminationGuardV842(
            strength=kwargs.get("contact_guard_strength", 0.85)
        )

    def config_dict(self) -> dict[str, object]:
        config = super().config_dict()
        config.update({
            "version": "v2.42",
            "ownership_owner": "TARGET_TOPOLOGY_CORE_SOURCE_HAIR_NEW_GROWTH",
            "matte_owner": "TARGET_CORE_TRANSITION_EVIDENCE",
            "face_guard": "TRANSITION_ONLY",
        })
        return config


__all__ = ["HairLocalRecompositionV842"]
