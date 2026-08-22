"""V2.41.2 stable-hue, gamut-safe, evaluation-aligned local recomposition."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.anchor_hair_evidence_v8411 import AnchorHairEvidenceV8411
from models.face_overlap_contamination_guard_v841 import FaceOverlapContaminationGuardV841
from models.hair_local_recomposition_v837 import HairLocalRecompositionV837
from models.hair_multiband_carrier_v840 import HairMultiBandCarrierV840
from models.reference_stable_hue_palette_v841 import ReferenceStableHuePaletteV841
from models.reference_tone_mapper_v840 import ReferenceToneMapperV840
from models.target_hair_matte_refiner_v841 import TargetHairMatteRefinerV841
from models.target_hair_occlusion_resolver_v8412 import TargetHairOcclusionResolverV8412
from models.target_stable_hue_chroma_field_v841 import TargetStableHueChromaFieldV841
from models.target_illumination_extractor_v8411 import TargetIlluminationExtractorV8411
from models.texture_preserving_recolor_v841 import TexturePreservingRecolorV841


class HairLocalRecompositionV841:
    def __init__(self, *, palette_mad_scale: float = 3.5, palette_min_support: int = 16,
                 illumination_radius: int = 11, carrier_mid_radius: int = 3,
                 mid_gain: float = 1.0, high_gain: float = 0.95,
                 chroma_mid_gain: float = 0.35, chroma_high_gain: float = 0.20,
                 contact_radius: int = 5,
                 matte_ring_radius: int = 5, contact_guard_strength: float = 0.85,
                 version: str = "v2.41.2") -> None:
        self.palette = ReferenceStableHuePaletteV841(
            mad_scale=palette_mad_scale, min_support=palette_min_support
        )
        self.carrier = HairMultiBandCarrierV840(low_radius=illumination_radius, mid_radius=carrier_mid_radius)
        self.target_illumination = TargetIlluminationExtractorV8411(low_radius=illumination_radius)
        self.tone_mapper = ReferenceToneMapperV840()
        self.relative_field = TargetStableHueChromaFieldV841(low_radius=illumination_radius)
        self.recolor = TexturePreservingRecolorV841(
            mid_gain=mid_gain, high_gain=high_gain,
            chroma_mid_gain=chroma_mid_gain, chroma_high_gain=chroma_high_gain,
        )
        self.resolver = TargetHairOcclusionResolverV8412(contact_radius=contact_radius)
        self.version = str(version)
        self.evidence = AnchorHairEvidenceV8411()
        self.matte = TargetHairMatteRefinerV841(ring_radius=matte_ring_radius)
        self.guard = FaceOverlapContaminationGuardV841(strength=contact_guard_strength)
        self.recomposer = HairLocalRecompositionV837(face_guard=0.0)

    def config_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "carrier": "V2.40_STRONG_ANCHOR_MULTIBAND",
            "hue_owner": "REFERENCE_STABLE_SINGLE_DIRECTION",
            "chroma_owner": "TARGET_L_ONLY_CHROMA_MAGNITUDE",
            "gamut_owner": "LOCAL_LAB_BINARY_SEARCH",
            "matte_owner": "ANCHOR_EVIDENCE_FACE_OVERLAP_REFINER",
            "face_guard": "FULL_FACE_SKIN_OVERLAP_SOFT_GUARD",
        }

    def __call__(self, *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                 color_reference_rgb: torch.Tensor, target_illumination_rgb: torch.Tensor,
                 coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
                 source_skin_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
                 source_hair_mask: torch.Tensor | None = None,
                 return_aux: bool = False):
        provisional_evidence, _ = self.evidence(
            allowed_hair=coarse_target_hair_mask, coarse_target_hair_mask=coarse_target_hair_mask,
            source_face_mask=source_face_mask, source_skin_mask=source_skin_mask,
            strong_anchor_rgb=strong_anchor_rgb, base_rgb=base_rgb, return_aux=True,
        )
        resolver_kwargs = dict(
            coarse_target_hair_mask=coarse_target_hair_mask,
            source_face_mask=source_face_mask, source_skin_mask=source_skin_mask,
            strong_anchor_rgb=strong_anchor_rgb, base_rgb=base_rgb,
            anchor_hair_evidence=provisional_evidence, return_aux=True,
        )
        if source_hair_mask is not None:
            resolver_kwargs["source_hair_mask"] = source_hair_mask
        allowed, ownership_aux = self.resolver(**resolver_kwargs)
        palette = self.palette(reference_rgb=color_reference_rgb, reference_hair_mask=reference_hair_mask)
        _, carrier_aux = self.carrier(
            strong_anchor_rgb=strong_anchor_rgb, base_rgb=base_rgb,
            allowed_hair_mask=allowed, hair_alpha_prior=allowed, return_aux=True,
        )
        illumination_aux = self.target_illumination(
            target_illumination_rgb=target_illumination_rgb,
            target_hair_mask=allowed, return_aux=True,
        )
        mapped_l_low, tone_aux = self.tone_mapper(
            target_l_low=illumination_aux["target_l_low"], hair_mask=allowed,
            reference_tone=palette, return_aux=True,
        )
        # ToneMapper already applies target low-frequency variation around the
        # reference center. Adding the residual again would double-count it.
        tone_aux["mapped_l_low"] = mapped_l_low
        target_residual = illumination_aux["target_l_residual"]
        mask_flat = allowed.flatten(1)
        target_flat, mapped_flat = target_residual.flatten(1), mapped_l_low.flatten(1)
        denom = mask_flat.sum(1).clamp_min(1.0)
        target_mean = (target_flat * mask_flat).sum(1) / denom
        mapped_mean = (mapped_flat * mask_flat).sum(1) / denom
        target_std = torch.sqrt((((target_flat - target_mean[:, None]) ** 2) * mask_flat).sum(1) / denom).clamp_min(1e-4)
        mapped_std = torch.sqrt((((mapped_flat - mapped_mean[:, None]) ** 2) * mask_flat).sum(1) / denom)
        illumination_gain = (mapped_std / target_std).view(-1, 1, 1, 1)
        field = self.relative_field(
            target_illumination_rgb=target_illumination_rgb, target_hair_mask=allowed,
            target_hair_alpha=allowed, palette=palette,
        )
        new_hair_rgb, recolor_aux = self.recolor(
            carrier_aux=carrier_aux, mapped_l_low=mapped_l_low,
            target_ab_low=field["target_ab_low"],
            hue_dispersion_deg=palette["hue_dispersion_deg"], return_aux=True,
        )
        hair_evidence, evidence_aux = self.evidence(
            allowed_hair=allowed, coarse_target_hair_mask=coarse_target_hair_mask,
            source_face_mask=source_face_mask, source_skin_mask=source_skin_mask,
            strong_anchor_rgb=strong_anchor_rgb, base_rgb=base_rgb,
            face_contact_ring=ownership_aux["face_contact_ring"], return_aux=True,
        )
        matte_kwargs = dict(
            base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb,
            new_hair_rgb=new_hair_rgb, coarse_target_hair_mask=coarse_target_hair_mask,
            occlusion_allowed_hair=allowed, source_face_mask=source_face_mask,
            source_skin_mask=source_skin_mask,
            face_contact_ring=ownership_aux["face_contact_ring"],
            face_intrusion_risk=ownership_aux["face_intrusion_risk"],
            anchor_hair_evidence=hair_evidence,
            anchor_texture_confidence=evidence_aux["anchor_texture_confidence"], return_aux=True,
        )
        matte_kwargs.update({key: ownership_aux[key] for key in ("target_hair_core", "target_hair_transition") if key in ownership_aux})
        alpha, matte_aux = self.matte(**matte_kwargs)
        guard_kwargs = dict(
            alpha=alpha, allowed_hair=allowed,
            coarse_target_hair_mask=coarse_target_hair_mask,
            source_face_mask=source_face_mask, source_skin_mask=source_skin_mask,
            strong_anchor_rgb=strong_anchor_rgb, base_rgb=base_rgb,
            new_hair_rgb=new_hair_rgb,
            face_contact_ring=ownership_aux["face_contact_ring"],
            face_intrusion_risk=ownership_aux["face_intrusion_risk"],
            hair_core=matte_aux["target_hair_core"],
            anchor_hair_evidence=hair_evidence, return_aux=True,
        )
        if "new_hair_core" in ownership_aux:
            guard_kwargs["new_hair_core"] = ownership_aux["new_hair_core"]
        guarded_alpha, guard_aux = self.guard(**guard_kwargs)
        # A second-stage gate only rejects pixels that are both uncertain hair
        # and high contamination risk; source face is never a blanket veto.
        confident_core = matte_aux["target_hair_core"] * (hair_evidence >= 0.65).float()
        reject_zone = matte_aux["target_hair_transition"] if "target_hair_transition" in matte_aux else torch.ones_like(confident_core)
        final_reject = (reject_zone * (guard_aux["combined_contamination_risk"] > 0.65).float() *
                        (guard_aux["anchor_hair_evidence"] < 0.35).float() *
                        (1.0 - confident_core)).clamp(0, 1)
        final_alpha = (guarded_alpha * (1.0 - final_reject)).clamp(0, 1)
        final_rgb = self.recomposer(
            base_rgb=base_rgb, new_hair_rgb=new_hair_rgb, hair_alpha=final_alpha,
            face_mask=torch.zeros_like(source_face_mask), target_hair_mask=torch.ones_like(final_alpha),
        )
        if not return_aux:
            return final_rgb
        aux = {
            **palette, **carrier_aux, **illumination_aux, **tone_aux, **field, **recolor_aux,
            **ownership_aux, **evidence_aux, **matte_aux, **guard_aux,
            "hair_ownership": allowed, "target_hair_alpha_final": final_alpha,
            "illumination_effective_gain": illumination_gain,
            "hair_alpha_raw": alpha,
            "target_hair_alpha_guarded": guarded_alpha,
            "final_face_overlap_reject": final_reject,
            "new_hair_rgb": new_hair_rgb, "final_rgb": final_rgb,
            "final_minus_base": final_rgb - base_rgb,
            "non_hair_mask": (1.0 - final_alpha).clamp(0, 1),
            "visible_face_mask": source_face_mask.float().clamp(0, 1) * (1.0 - final_alpha),
            "visible_skin_mask": source_skin_mask.float().clamp(0, 1) * (1.0 - final_alpha),
            "deep_uncertain_skin_overlap": ownership_aux["deep_overlap"] * (hair_evidence < 0.65).float(),
            "source_face_boundary_ring": (F.max_pool2d(source_face_mask.float(), 7, stride=1, padding=3) -
                                           (-F.max_pool2d(-source_face_mask.float(), 7, stride=1, padding=3))).clamp(0, 1),
            "non_hair_leakage_map": (final_rgb - base_rgb).abs().mean(1, keepdim=True) * (1.0 - final_alpha),
            "face_rgb_change_from_base": ((final_rgb - base_rgb) * source_face_mask).abs().amax(dim=(1, 2, 3)),
            "non_hair_rgb_change_from_base": ((final_rgb - base_rgb) * (1.0 - final_alpha)).abs().amax(dim=(1, 2, 3)),
            "chroma_strength": (recolor_aux["new_chroma"] / recolor_aux["reference_chroma_low"].clamp_min(1e-4)).clamp(0, 2),
        }
        alpha_local_range = F.max_pool2d(final_alpha, 7, stride=1, padding=3) - (-F.max_pool2d(-final_alpha, 7, stride=1, padding=3))
        aux["face_boundary_alpha_jump_map"] = alpha_local_range * aux["source_face_boundary_ring"] * coarse_target_hair_mask.float().clamp(0, 1)
        final_lab = rgb_to_lab(final_rgb)
        final_c = torch.linalg.vector_norm(final_lab[:, 1:], dim=1, keepdim=True)
        target_c = recolor_aux["reference_chroma_low"]
        chroma_under = matte_aux["target_hair_core"] * (target_c >= 5.0).float() * ((final_c / target_c.clamp_min(1e-4)) < 0.60).float()
        low_chroma_under = matte_aux["target_hair_core"] * (target_c < 5.0).float() * ((final_lab[:, :1] - recolor_aux["final_l_safe"]).abs() > 8.0).float()
        aux["hair_undertransfer_map"] = torch.maximum(chroma_under, low_chroma_under)
        return final_rgb, aux


__all__ = ["HairLocalRecompositionV841"]
