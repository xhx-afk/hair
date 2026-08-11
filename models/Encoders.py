import argparse

import clip
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential
from torchvision import transforms as T

from models.Net import FeatureEncoderMult, IBasicBlock, conv1x1
from models.color_condition_v8 import COLOR_DESCRIPTOR_DIM
from models.stylegan2.model import PixelNorm


class ModulationModule(nn.Module):
    def __init__(self, layernum, last=False, inp=512, middle=512):
        super().__init__()
        self.layernum = layernum
        self.last = last
        self.fc = Linear(512, 512)
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = Sequential(Linear(inp, middle), LayerNorm([middle]), LeakyReLU(), Linear(middle, 512))
        self.beta_function = Sequential(Linear(inp, middle), LayerNorm([middle]), LeakyReLU(), Linear(middle, 512))
        self.leakyrelu = LeakyReLU()

    def forward(self, x, embedding):
        x = self.fc(x)
        x = self.norm(x)
        gamma = self.gamma_function(embedding)
        beta = self.beta_function(embedding)
        out = x * (1 + gamma) + beta
        if not self.last:
            out = self.leakyrelu(out)
        return out


class FeatureiResnet(nn.Module):
    def __init__(self, blocks, inplanes=1024):
        super().__init__()

        self.res_blocks = {}

        for n, block in enumerate(blocks, start=1):
            planes, num_blocks = block

            for k in range(1, num_blocks + 1):
                downsample = None
                if inplanes != planes:
                    downsample = nn.Sequential(conv1x1(inplanes, planes, 1), nn.BatchNorm2d(planes, eps=1e-05, ), )

                self.res_blocks[f'res_block_{n}_{k}'] = IBasicBlock(inplanes, planes, 1, downsample, 1, 64, 1)
                inplanes = planes

        self.res_blocks = nn.ModuleDict(self.res_blocks)

    def forward(self, x):
        for module in self.res_blocks.values():
            x = module(x)
        return x


class RotateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(6, i == 4) for i in range(5)])

    def forward(self, latent_from, latent_to):
        dt_latent = self.pixelnorm(latent_from)
        for modulation_module in self.modulation_module_list:
            dt_latent = modulation_module(dt_latent, latent_to)
        output = latent_from + 0.1 * dt_latent
        return output


class ClipBlendingModel(nn.Module):
    def __init__(self, clip_model="ViT-B/32"):
        super().__init__()
        self.pixelnorm = PixelNorm()
        self.clip_model, _ = clip.load(clip_model, device="cuda")
        self.transform = T.Compose(
            [T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.modulation_module_list = nn.ModuleList(
            [ModulationModule(12, i == 4, inp=512 * 3, middle=1024) for i in range(5)]
        )

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def get_image_embed(self, image_tensor):
        resized_tensor = self.face_pool(image_tensor)
        renormed_tensor = self.transform(resized_tensor * 0.5 + 0.5)
        return self.clip_model.encode_image(renormed_tensor)

    def forward(self, latent_face, latent_color, target_face, hair_color):
        embed_face = self.get_image_embed(target_face).unsqueeze(1).expand(-1, 12, -1)
        embed_color = self.get_image_embed(hair_color).unsqueeze(1).expand(-1, 12, -1)
        latent_in = torch.cat((latent_color, embed_face, embed_color), dim=-1)

        dt_latent = self.pixelnorm(latent_face)
        for modulation_module in self.modulation_module_list:
            dt_latent = modulation_module(dt_latent, latent_in)
        output = latent_face + 0.1 * dt_latent
        return output


DIRECT_COLOR_ARCH_V8_2 = "direct_color_anchor_v8_2"


def load_direct_color_adapter_state_v8(module, state_dict):
    """Strictly load every V2 adapter tensor except the frozen CLIP weights."""
    incompatible = module.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("clip_model.")
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Invalid direct_color_anchor_v8_2 state_dict: "
            f"missing_non_clip={invalid_missing}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    return {
        "loaded": sorted(state_dict.keys()),
        "missing_clip": sorted(incompatible.missing_keys),
    }


class DirectColorBlendAdapterV8(nn.Module):
    def __init__(
        self,
        clip_model="ViT-B/32",
        descriptor_dim=COLOR_DESCRIPTOR_DIM,
        direct_mix_init=0.65,
        direct_mix_floor=0.20,
        correction_budget_ratio=0.25,
        clip_model_module=None,
        descriptor_hidden=128,
        correction_hidden=512,
        layer_embed_dim=64,
    ):
        super().__init__()
        self.pixelnorm = PixelNorm()
        if clip_model_module is None:
            self.clip_model, _ = clip.load(clip_model, device="cuda")
        else:
            self.clip_model = clip_model_module
        self.transform = T.Compose(
            [T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
        )
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, descriptor_hidden),
            nn.LayerNorm(descriptor_hidden),
            nn.LeakyReLU(),
            nn.Linear(descriptor_hidden, descriptor_hidden),
            nn.LayerNorm(descriptor_hidden),
            nn.LeakyReLU(),
        )
        self.layer_mix_head = nn.Linear(descriptor_hidden, 12)
        self.layer_embedding = nn.Embedding(12, layer_embed_dim)
        correction_input_dim = 512 + 512 + descriptor_hidden + layer_embed_dim
        self.correction_mlp = nn.Sequential(
            nn.Linear(correction_input_dim, correction_hidden),
            nn.LayerNorm(correction_hidden),
            nn.LeakyReLU(),
            nn.Linear(correction_hidden, correction_hidden),
            nn.LayerNorm(correction_hidden),
            nn.LeakyReLU(),
            nn.Linear(correction_hidden, 512),
        )
        if not 0.0 < direct_mix_init < 1.0:
            raise ValueError("direct_mix_init must be strictly between 0 and 1")
        if not 0.0 <= direct_mix_floor <= 1.0:
            raise ValueError("direct_mix_floor must be in [0,1]")
        if correction_budget_ratio < 0.0:
            raise ValueError("correction_budget_ratio must be non-negative")
        self.direct_mix_init = float(direct_mix_init)
        self.direct_mix_floor = float(direct_mix_floor)
        self.correction_budget_ratio = float(correction_budget_ratio)

        nn.init.zeros_(self.layer_mix_head.weight)
        init_logit = torch.logit(torch.tensor(self.direct_mix_init)).item()
        nn.init.constant_(self.layer_mix_head.bias, init_logit)
        nn.init.zeros_(self.correction_mlp[-1].weight)
        nn.init.zeros_(self.correction_mlp[-1].bias)

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def get_image_embed(self, image_tensor):
        resized_tensor = self.face_pool(image_tensor)
        renormed_tensor = self.transform(resized_tensor * 0.5 + 0.5)
        return self.clip_model.encode_image(renormed_tensor).float()

    @staticmethod
    def _gate_3d(gate_value, batch, device, dtype, name):
        gate = torch.as_tensor(gate_value, device=device, dtype=dtype)
        if gate.numel() != batch:
            raise ValueError(f"Expected {batch} {name} values, got shape={tuple(gate.shape)}")
        return gate.reshape(batch, 1, 1).clamp(0, 1)

    def set_correction_trainable(self, enabled):
        for module in (self.layer_embedding, self.correction_mlp):
            for parameter in module.parameters():
                parameter.requires_grad = bool(enabled)

    @staticmethod
    def _layer_mix_override(value, batch, device, dtype):
        mix = torch.as_tensor(value, device=device, dtype=dtype)
        if mix.numel() == 1:
            return mix.reshape(1, 1, 1).expand(batch, 12, 1).clamp(0, 1)
        if mix.shape == (batch, 12):
            mix = mix.unsqueeze(-1)
        if mix.shape != (batch, 12, 1):
            raise ValueError(f"layer_mix_override must be scalar, [B,12], or [B,12,1], got {tuple(mix.shape)}")
        return mix.clamp(0, 1)

    def forward(
        self,
        latent_face,
        latent_color,
        color_descriptor,
        chroma_need_gate,
        lightness_need_gate,
        edit_need_gate,
        correction_enabled=True,
        layer_mix_override=None,
        return_aux=False,
    ):
        if latent_face.shape != latent_color.shape or latent_face.shape[1:] != (12, 512):
            raise ValueError(
                "DirectColorBlendAdapterV8 expects matching [B,12,512] face/color latents, "
                f"got {tuple(latent_face.shape)} and {tuple(latent_color.shape)}"
            )
        batch = latent_face.size(0)
        chroma_gate = self._gate_3d(
            chroma_need_gate, batch, latent_face.device, latent_face.dtype, "chroma gate"
        )
        lightness_gate = self._gate_3d(
            lightness_need_gate, batch, latent_face.device, latent_face.dtype, "lightness gate"
        )
        edit_gate = self._gate_3d(
            edit_need_gate, batch, latent_face.device, latent_face.dtype, "edit gate"
        )
        if color_descriptor.shape != (batch, COLOR_DESCRIPTOR_DIM):
            raise ValueError(
                f"Expected color descriptor [{batch},{COLOR_DESCRIPTOR_DIM}], "
                f"got {tuple(color_descriptor.shape)}"
            )

        if not torch.isfinite(color_descriptor).all():
            raise ValueError("color_descriptor contains NaN or Inf")
        descriptor_feature = self.descriptor_encoder(color_descriptor.float()).to(latent_face.dtype)
        learned_mix = torch.sigmoid(self.layer_mix_head(descriptor_feature)).unsqueeze(-1)
        mix_floor = self.direct_mix_floor * edit_gate
        layer_mix = mix_floor + (1.0 - mix_floor) * learned_mix
        if layer_mix_override is not None:
            layer_mix = self._layer_mix_override(
                layer_mix_override, batch, latent_face.device, latent_face.dtype
            )
        direct_delta = latent_color - latent_face
        direct_component = edit_gate * layer_mix * direct_delta

        if correction_enabled:
            source_normalized = self.pixelnorm(latent_face)
            delta_normalized = self.pixelnorm(direct_delta)
            descriptor_layers = descriptor_feature[:, None, :].expand(-1, 12, -1)
            layer_indices = torch.arange(12, device=latent_face.device)
            layer_feature = self.layer_embedding(layer_indices)[None].expand(batch, -1, -1)
            correction_input = torch.cat(
                (source_normalized, delta_normalized, descriptor_layers, layer_feature),
                dim=-1,
            )
            raw_correction = self.correction_mlp(correction_input)
        else:
            raw_correction = torch.zeros_like(direct_delta)

        direct_delta_norm = torch.linalg.vector_norm(direct_delta.flatten(1), dim=1)
        direct_component_norm = torch.linalg.vector_norm(direct_component.flatten(1), dim=1)
        correction_raw_norm = torch.linalg.vector_norm(raw_correction.flatten(1), dim=1)
        correction_budget = self.correction_budget_ratio * direct_component_norm.detach()
        requested_correction_scale = correction_budget / (correction_raw_norm + 1e-6)
        correction_budget_scale = torch.where(
            requested_correction_scale < 1.0,
            requested_correction_scale * (1.0 - 1e-5),
            torch.ones_like(requested_correction_scale),
        )
        correction = raw_correction * correction_budget_scale[:, None, None]
        output = latent_face + direct_component + edit_gate * correction

        assert direct_delta.shape == (batch, 12, 512)
        assert layer_mix.shape == (batch, 12, 1)
        assert output.shape == (batch, 12, 512)
        assert torch.isfinite(output).all()
        assert bool(((edit_gate >= 0) & (edit_gate <= 1)).all())
        assert bool(((layer_mix >= 0) & (layer_mix <= 1)).all())
        if output.shape != (batch, 12, 512):
            raise RuntimeError(f"BlendingV8 output contract broken: shape={tuple(output.shape)}")

        if not return_aux:
            return output
        correction_norm = torch.linalg.vector_norm(correction.flatten(1), dim=1)
        total_delta_norm = torch.linalg.vector_norm((output - latent_face).flatten(1), dim=1)
        return output, {
            "direct_delta": direct_delta,
            "direct_component": direct_component,
            "correction": correction,
            "direct_delta_norm": direct_delta_norm,
            "direct_component_norm": direct_component_norm,
            "correction_raw_norm": correction_raw_norm,
            "correction_norm": correction_norm,
            "correction_budget": correction_budget,
            "correction_budget_scale": correction_budget_scale,
            "layer_mix": layer_mix,
            "layer_mix_mean": layer_mix.mean(dim=(1, 2)),
            "layer_mix_min": layer_mix.amin(dim=(1, 2)),
            "layer_mix_max": layer_mix.amax(dim=(1, 2)),
            "chroma_need_gate": chroma_gate.flatten(),
            "lightness_need_gate": lightness_gate.flatten(),
            "edit_need_gate": edit_gate.flatten(),
            "direct_mix_fraction": direct_component_norm / direct_delta_norm.clamp_min(1e-6),
            "correction_to_direct_ratio": correction_norm / direct_component_norm.clamp_min(1e-6),
            "total_delta_norm": total_delta_norm,
            "total_to_direct_ratio": total_delta_norm / direct_component_norm.clamp_min(1e-6),
        }


class PostProcessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_face = FeatureEncoderMult(fs_layers=[9], opts=argparse.Namespace(
            **{'arcface_model_path': "pretrained_models/ArcFace/backbone_ir50.pth"}))

        self.latent_avg = torch.load('pretrained_models/PostProcess/latent_avg.pt', map_location=torch.device('cuda'))
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])

        self.to_latent_1 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
        self.to_latent_2 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
        self.pixelnorm = PixelNorm()

    def forward(self, source, target):
        s_face, [f_face] = self.encoder_face(source)
        s_hair, [f_hair] = self.encoder_face(target)

        dt_latent_face = self.pixelnorm(s_face)
        dt_latent_hair = self.pixelnorm(s_hair)

        for mod_module in self.to_latent_1:
            dt_latent_face = mod_module(dt_latent_face, s_hair)

        for mod_module in self.to_latent_2:
            dt_latent_hair = mod_module(dt_latent_hair, s_face)

        finall_s = self.latent_avg + 0.1 * (dt_latent_face + dt_latent_hair)

        cat_f = torch.cat((f_face, f_hair), dim=1)
        finall_f = self.to_feature(cat_f)

        return finall_s, finall_f


class ClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.transform = T.Compose(
            [T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
        )
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, image_tensor):
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.to("cuda")
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor / 255

        resized_tensor = self.face_pool(image_tensor)
        renormed_tensor = self.transform(resized_tensor)
        return self.clip_model.encode_image(renormed_tensor)
