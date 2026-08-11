import torch

from models.Blending import Blending
from models.Encoders import (
    DIRECT_COLOR_ARCH_V8_2,
    DirectColorBlendAdapterV8,
    PostProcessModel,
    load_direct_color_adapter_state_v8,
)
from models.Net import Net
from models.color_condition_v8 import ColorConditionConfigV8, build_color_condition_bundle
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents


class Blending_v8(Blending):
    """
    v8 keeps the stable blending branch, but makes the implementation
    self-contained so it does not depend on Blending_v4 from the server.
    """

    def __init__(self, opts, net=None):
        torch.nn.Module.__init__(self)
        self.opts = opts
        self.net = Net(self.opts) if net is None else net

        checkpoint = torch.load(self.opts.blending_checkpoint, map_location=self.opts.device)
        checkpoint_arch = checkpoint.get("arch") if isinstance(checkpoint, dict) else None
        if checkpoint_arch != DIRECT_COLOR_ARCH_V8_2:
            raise RuntimeError(
                f"Refusing incompatible BlendingV8 checkpoint {self.opts.blending_checkpoint}: "
                f"arch={checkpoint_arch!r}, required={DIRECT_COLOR_ARCH_V8_2!r}"
            )
        adapter_config = checkpoint.get("adapter_config", {})
        self.blending_encoder = DirectColorBlendAdapterV8(
            checkpoint.get("clip", "ViT-B/32"),
            direct_mix_init=adapter_config.get(
                "direct_mix_init", getattr(self.opts, "direct_mix_init_v8", 0.65)
            ),
            direct_mix_floor=adapter_config.get(
                "direct_mix_floor", getattr(self.opts, "direct_mix_floor_v8", 0.20)
            ),
            correction_budget_ratio=adapter_config.get(
                "correction_budget_ratio", getattr(self.opts, "correction_budget_ratio_v8", 0.25)
            ),
        )
        source_state = checkpoint.get("model_state_dict", checkpoint)
        report = load_direct_color_adapter_state_v8(self.blending_encoder, source_state)
        print(
            f"[Blending_v8] loaded arch={DIRECT_COLOR_ARCH_V8_2} "
            f"strict adapter tensors={len(report['loaded'])}"
        )
        self.blending_encoder.to(self.opts.device).eval()

        self.post_process = PostProcessModel().to(self.opts.device).eval()
        postprocess_checkpoint = torch.load(self.opts.pp_checkpoint, map_location=self.opts.device)
        self.post_process.load_state_dict(postprocess_checkpoint["model_state_dict"])
        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.color_config = ColorConditionConfigV8(
            chroma_no_edit_threshold=getattr(self.opts, "chroma_no_edit_threshold_v8", 3.0),
            chroma_full_edit_threshold=getattr(self.opts, "chroma_full_edit_threshold_v8", 19.0),
            lightness_no_edit_threshold=getattr(self.opts, "lightness_no_edit_threshold_v8", 3.0),
            lightness_full_edit_threshold=getattr(self.opts, "lightness_full_edit_threshold_v8", 15.0),
            max_global_l_shift=getattr(self.opts, "max_global_l_shift_v8", 20.0),
            min_safe_fraction=getattr(self.opts, "min_safe_reference_fraction_v8", 0.35),
        )

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        del align_color
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_2 = name_to_embed["shape"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        HM_3 = torch.where(color_mask == 13, torch.ones_like(color_mask), torch.zeros_like(color_mask)).float()
        _, HM_3E = self.dilate_erosion.mask(HM_3)
        hair_color_mask = HM_3E

        latent_S_1 = name_to_embed["face"]["S"]
        latent_S_3 = name_to_embed["color"]["S"]
        latent_F_align = align_shape["latent_F_align"]
        HM_X = align_shape["HM_X"]

        _, HM_XE = self.dilate_erosion.mask(HM_X)

        if I_1 is not I_3 or I_1 is not I_2:
            I_base, _ = self.net.generator(
                [latent_S_1],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_align,
            )
            I_base_256 = self.downsample_256(I_base)
            bundle = build_color_condition_bundle(
                reference_image=I_3,
                reference_hair_mask=hair_color_mask,
                base_image=I_base_256,
                target_hair_mask=HM_XE,
                config=self.color_config,
            )
            S_blend_6_18, blending_aux = self.blending_encoder(
                latent_face=latent_S_1[:, 6:],
                latent_color=latent_S_3[:, 6:],
                color_descriptor=bundle["descriptor"],
                chroma_need_gate=bundle["chroma_need_gate"],
                lightness_need_gate=bundle["lightness_need_gate"],
                edit_need_gate=bundle["edit_need_gate"],
                correction_enabled=True,
                return_aux=True,
            )
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1
            bundle = None
            blending_aux = None

        I_blend, _ = self.net.generator(
            [S_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_align,
        )
        I_blend_256 = self.downsample_256(I_blend)

        S_final, F_final = self.post_process(I_1, I_blend_256)
        I_final, _ = self.net.generator(
            [S_final],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=F_final,
        )

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name")
            exp_name = exp_name if exp_name is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending_v8", "blending.png", I_blend)
            save_latents(output_dir, "Blending_v8", "blending.npz", S_blend=S_blend)
            if bundle is not None:
                save_gen_image(output_dir, "Blending_v8", "base.png", I_base_256)
                save_gen_image(output_dir, "Blending_v8", "color_proxy.png", bundle["color_proxy"])
                save_gen_image(output_dir, "Blending_v8", "pseudo_color.png", bundle["pseudo_rgb"])
                save_latents(
                    output_dir,
                    "Blending_v8",
                    "color_condition.npz",
                    descriptor=bundle["descriptor"],
                    chroma_need_gate=bundle["chroma_need_gate"],
                    lightness_need_gate=bundle["lightness_need_gate"],
                    edit_need_gate=bundle["edit_need_gate"],
                    safe_ref_mask=bundle["safe_ref_mask"],
                    rejected_highlight_mask=bundle["rejected_highlight_mask"],
                    direct_delta_norm=blending_aux["direct_delta_norm"],
                    direct_component_norm=blending_aux["direct_component_norm"],
                    layer_mix=blending_aux["layer_mix"],
                    correction_norm=blending_aux["correction_norm"],
                    correction_budget=blending_aux["correction_budget"],
                    total_delta_norm=blending_aux["total_delta_norm"],
                )
            save_gen_image(output_dir, "Final_v8", "final.png", I_final)
            save_latents(output_dir, "Final_v8", "final.npz", S_final=S_final, F_final=F_final)

        return ((I_final[0] + 1) / 2).clamp(0, 1)
