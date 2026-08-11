import atexit
import gc
import json
import multiprocessing
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tnf
from PIL import Image
from joblib.externals.loky import get_reusable_executor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v8 import HairFast_v8, get_parser_v8
from models.Encoders import (
    DIRECT_COLOR_ARCH_V8_2,
    DirectColorBlendAdapterV8 as BlendingModel,
    load_direct_color_adapter_state_v8,
)
from models.Net import Net
from models.SG_IDCT_v16 import gaussian_blur2d, rgb_to_lab
from models.color_condition_v8 import ColorConditionConfigV8, build_color_condition_bundle
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_latents
from utils.train import get_fid_calc, toggle_grad


def clean_zombies():
    try:
        get_reusable_executor().shutdown(wait=False, kill_workers=True)
    except Exception as exc:
        print(f"[blending_v8] loky cleanup skipped: {exc}", file=sys.stderr)

    for process in multiprocessing.active_children():
        try:
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        except Exception as exc:
            pid = getattr(process, "pid", "unknown")
            print(f"[blending_v8] child cleanup skipped for pid={pid}: {exc}", file=sys.stderr)


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = os.environ.get("BLENDING_V8_DATASET_PROFILE", "small").strip().lower()

USER_DATASET_DIR_FFHQ = Path("input/blending_dataset_v8")
USER_FACE_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/blending_train_v8_direct_anchor_v2")
USER_VAL_SIZE_FFHQ = 512

USER_DATASET_DIR_SMALL = Path("input/blending_dataset_v8_small_v2_short_to_long")
USER_FACE_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")
USER_SHAPE_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/long")
USER_COLOR_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_color")
USER_OUTPUT_DIR_SMALL = Path("output/blending_train_v8_direct_anchor_v2_small")
USER_VAL_SIZE_SMALL = 64

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_BATCH_SIZE = 16
USER_GRAD_ACCUM_STEPS = 1  # effective batch size = USER_BATCH_SIZE * USER_GRAD_ACCUM_STEPS
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_EPOCHS = 40
USER_STAGE_A_EPOCHS = 15
USER_LR_STAGE_A = 1e-4
USER_LR_STAGE_B = 5e-5
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 5.0
USER_FACE_CLIP_LOSS_WEIGHT = 1.0
USER_PSEUDO_AB_LOSS_WEIGHT = 12.0
USER_PSEUDO_RGB_LOSS_WEIGHT = 1.0
USER_PSEUDO_LUMA_LOSS_WEIGHT = 3.0
USER_POSITIVE_LUMA_EXCESS_WEIGHT = 4.0
USER_HF_LUMA_EXCESS_WEIGHT = 1.0
USER_CORRECTION_NORM_WEIGHT = 0.10
USER_LUMA_EXCESS_MARGIN = 4.0
USER_HF_LUMA_EXCESS_MARGIN = 1.5
USER_FACE_KEEP_L1_LOSS_WEIGHT = 1.0
USER_REMOVE_KEEP_L1_LOSS_WEIGHT = 1.1
USER_PROTECT_CHROMA_KEEP_LOSS_WEIGHT = 2.5
USER_SKIN_CHROMA_KEEP_LOSS_WEIGHT = 8.0
USER_SKIN_RGB_KEEP_LOSS_WEIGHT = 4.0
USER_SAFE_HAIR_MIN_PIXELS = 64.0
USER_REMOVE_BLOCK_IN_TARGET_HAIR = 0.12
USER_FACE_NECK_COLOR_BLOCK = 0.96
USER_TARGET_HAIR_NECK_OVERRIDE = 0.94
USER_AUTHOR_COLOR_ALIGN_BATCH_PROB = 0.0
USER_AUTHOR_ZERO_PREFIX_TRAIN = True

USER_CHROMA_NO_EDIT_THRESHOLD_V8 = 3.0
USER_CHROMA_FULL_EDIT_THRESHOLD_V8 = 19.0
USER_LIGHTNESS_NO_EDIT_THRESHOLD_V8 = 3.0
USER_LIGHTNESS_FULL_EDIT_THRESHOLD_V8 = 15.0
USER_MAX_GLOBAL_L_SHIFT_V8 = 20.0
USER_MIN_SAFE_REFERENCE_FRACTION_V8 = 0.35
USER_HIGHLIGHT_MAD_SCALE = 1.8
USER_HIGHLIGHT_GLOBAL_MIN_MARGIN = 3.0
USER_HIGHLIGHT_LOCAL_L_MARGIN = 2.5
USER_HIGHLIGHT_LOCAL_C_MARGIN = 1.5
USER_HIGHLIGHT_CHROMA_RATIO = 0.82
USER_DIRECT_MIX_INIT = 0.65
USER_DIRECT_MIX_FLOOR = 0.20
USER_CORRECTION_BUDGET_RATIO = 0.25
USER_CLIP_MODEL = "ViT-B/32"
USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "/data/coding/HairFastGAN/HairFastGAN-main/best.pth"
USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0
USER_BUILD_CACHE_WITH_CURRENT_SATD = False
USER_FORCE_REFRESH_ALIGN_CACHE = False

USER_USE_FID = False
USER_FID_CACHE = "input/fid.pkl"
USER_FID_DATASET = Path("images/FFHQ")

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_PREVIEW_EVERY = 1
USER_LOG_IMAGE_COUNT = 30
# Set this to output/.../checkpoints/last.pth or best.pth to continue training.
USER_RESUME_CHECKPOINT = ""

# Shape/SATD remains the validation and inference geometry. During training,
# a minority of batches use the author's face->color alignment so the encoder
# also learns reference color across the complete reference-hair extent.
# ============================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        },
        "small": {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "val_size": USER_VAL_SIZE_SMALL,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(
            f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}. "
            f"Choose one of: {', '.join(sorted(profiles))}."
        )
    return profiles[USER_DATASET_PROFILE]


PROFILE = resolve_dataset_profile()
ACTIVE_DATASET_DIR = PROFILE["dataset_dir"]
ACTIVE_FACE_ROOT = PROFILE["face_root"]
ACTIVE_SHAPE_ROOT = PROFILE["shape_root"]
ACTIVE_COLOR_ROOT = PROFILE["color_root"]
ACTIVE_OUTPUT_DIR = PROFILE["output_dir"]
ACTIVE_VAL_SIZE = PROFILE["val_size"]

if USER_BATCH_SIZE < 1:
    raise RuntimeError("USER_BATCH_SIZE must be >= 1.")
if USER_GRAD_ACCUM_STEPS < 1:
    raise RuntimeError("USER_GRAD_ACCUM_STEPS must be >= 1.")
if USER_AUTHOR_COLOR_ALIGN_BATCH_PROB != 0.0:
    raise RuntimeError("BlendingV8 color-direction fix requires USER_AUTHOR_COLOR_ALIGN_BATCH_PROB=0.0.")
if not 0.0 <= USER_TARGET_HAIR_NECK_OVERRIDE <= 1.0:
    raise RuntimeError("USER_TARGET_HAIR_NECK_OVERRIDE must be in [0, 1].")
if not 0 <= USER_STAGE_A_EPOCHS <= USER_EPOCHS:
    raise RuntimeError("USER_STAGE_A_EPOCHS must be in [0, USER_EPOCHS].")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_image_path(root: Path, stem: str) -> Path:
    png_path = root / f"{stem}.png"
    if png_path.exists():
        return png_path
    jpg_path = root / f"{stem}.jpg"
    if jpg_path.exists():
        return jpg_path
    jpeg_path = root / f"{stem}.jpeg"
    if jpeg_path.exists():
        return jpeg_path
    raise FileNotFoundError(f"Cannot find {stem}.png/.jpg/.jpeg in {root}")


def read_triplets(dataset_dir: Path) -> list[tuple[str, str, str]]:
    triplets = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as handle:
        for line in handle:
            items = line.strip().split()
            if len(items) == 3:
                triplets.append((items[0], items[1], items[2]))
    return triplets


def identity_func(align_shape, align_color, name_to_embed, **kwargs):
    return align_shape, align_color, name_to_embed


def role_key(role: str, stem: str) -> str:
    return f"{role}__{stem}"


def fs_cache_name(role: str, stem: str) -> str:
    return f"{role_key(role, stem)}.npz"


def align_cache_name(face_name: str, ref_role: str, ref_name: str) -> str:
    return f"{role_key('face', face_name)}_{role_key(ref_role, ref_name)}.npz"


def build_remove_protect_mask(align_info: dict[str, object]) -> torch.Tensor:
    delta_masks = align_info.get("delta_masks")
    if not isinstance(delta_masks, dict):
        hm_x = align_info["HM_X"]
        return torch.zeros_like(hm_x).float()

    remove = delta_masks["M_remove"].float()
    zero = torch.zeros_like(remove)
    protect = (
        1.00 * remove
        + 0.95 * delta_masks.get("M_remove_halo", zero).float()
        + 0.88 * delta_masks.get("M_remove_face", zero).float()
        + 0.92 * delta_masks.get("M_remove_neck", zero).float()
        + 0.92 * delta_masks.get("M_remove_tail", zero).float()
        + 0.70 * delta_masks.get("M_face_strand_probe", zero).float()
        + 0.80 * delta_masks.get("M_remove_context", zero).float()
        + 0.86 * delta_masks.get("M_body_preserve", zero).float()
        + 0.72 * delta_masks.get("M_visible_body_anchor", zero).float()
        + 0.60 * delta_masks.get("M_body_region", zero).float()
        + 0.68 * delta_masks.get("M_cloth_region", zero).float()
        + 0.35 * delta_masks.get("M_boundary", zero).float()
    )
    return protect.clamp(0, 1)


def align_instead_shape(hair_fast):
    def shape_module(func):
        def wrapper(*args, **kwargs):
            if kwargs.get("align_flag", False):
                return hair_fast.align.align_images(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    def align_module(func):
        def wrapper(*args, **kwargs):
            if "align_flag" in kwargs:
                kwargs = kwargs.copy()
                kwargs.pop("align_flag")
            return func(*args, **kwargs)

        return wrapper

    hair_fast.align.shape_module = shape_module(hair_fast.align.shape_module)
    hair_fast.align.align_images = align_module(hair_fast.align.align_images)


def build_cache_model() -> HairFast_v8:
    model_args = get_parser_v8().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.use_satd_v8 = bool(USER_USE_SATD_V8)
    model_args.satd_checkpoint_v8 = USER_SATD_CHECKPOINT_V8
    model_args.satd_blend_v8 = USER_SATD_BLEND_V8
    model_args.satd_boundary_v8 = USER_SATD_BOUNDARY_V8
    model_args.eq8_reference_blend_v8 = USER_EQ8_REFERENCE_BLEND_V8

    hair_fast = HairFast_v8(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)
    return hair_fast


def ensure_dataset_cache_v8(triplets: list[tuple[str, str, str]]):
    if USER_USE_SATD_V8:
        if not USER_SATD_CHECKPOINT_V8:
            raise RuntimeError("USER_SATD_CHECKPOINT_V8 is empty while USER_USE_SATD_V8=True.")
        if not Path(USER_SATD_CHECKPOINT_V8).exists():
            raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V8: {USER_SATD_CHECKPOINT_V8}")

    fs_dir = ACTIVE_DATASET_DIR / "FS"
    align_dir = ACTIVE_DATASET_DIR / "Align"
    mask_dir = ACTIVE_DATASET_DIR / "Masks"
    fs_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    def fs_path(role: str, stem: str) -> Path:
        return fs_dir / fs_cache_name(role, stem)

    def align_path(face_stem: str, ref_role: str, ref_stem: str) -> Path:
        return align_dir / align_cache_name(face_stem, ref_role, ref_stem)

    def mask_path(face_stem: str, ref_role: str, ref_stem: str) -> Path:
        return mask_dir / align_cache_name(face_stem, ref_role, ref_stem)

    def mask_has_target_hair(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with np.load(path) as data:
                return "target_hair" in data.files
        except Exception:
            return False

    def missing_required_cache() -> list[Path]:
        missing = []
        for face_name, shape_name, color_name in triplets:
            shape_mask_path = mask_path(face_name, "shape", shape_name)
            color_mask_path = mask_path(face_name, "color", color_name)
            required = [
                fs_path("face", face_name),
                fs_path("shape", shape_name),
                fs_path("color", color_name),
                align_path(face_name, "shape", shape_name),
                align_path(face_name, "color", color_name),
            ]
            missing.extend(path for path in required if not path.exists())
            if not mask_has_target_hair(shape_mask_path):
                missing.append(shape_mask_path)
            if not mask_has_target_hair(color_mask_path):
                missing.append(color_mask_path)
        return missing

    if not USER_BUILD_CACHE_WITH_CURRENT_SATD:
        missing = missing_required_cache()
        if missing:
            preview = "\n".join(f"  {path}" for path in missing[:10])
            raise RuntimeError(
                "Role-scoped v8 blending cache is missing or stale. The Masks cache must include "
                "target_hair for long-hair color supervision, and the old unscoped cache can mix "
                "FFHQ_long/FFHQ_short/FFHQ_color entries with the same stem. "
                "Run scripts/blending_gen_v8.py again or set USER_BUILD_CACHE_WITH_CURRENT_SATD=True "
                f"to rebuild it.\nMissing examples:\n{preview}"
            )
        return

    hair_fast = build_cache_model()
    required_triplets = []
    for face_name, shape_name, color_name in triplets:
        need_align_shape = USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, "shape", shape_name).exists())
        need_align_color = USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, "color", color_name).exists())
        need_mask_shape = USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_has_target_hair(mask_path(face_name, "shape", shape_name)))
        need_mask_color = USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_has_target_hair(mask_path(face_name, "color", color_name)))
        need_face_fs = not fs_path("face", face_name).exists()
        need_shape_fs = not fs_path("shape", shape_name).exists()
        need_color_fs = not fs_path("color", color_name).exists()
        if need_align_shape or need_align_color or need_mask_shape or need_mask_color or need_face_fs or need_shape_fs or need_color_fs:
            required_triplets.append((face_name, shape_name, color_name))

    if not required_triplets:
        print("[blending_v8] FS/Align cache already matches current training needs.", file=sys.stderr)
        return

    print(
        f"[blending_v8] rebuilding cache for {len(required_triplets)} triplets "
        f"(use_satd_v8={USER_USE_SATD_V8}, satd_checkpoint={USER_SATD_CHECKPOINT_V8})",
        file=sys.stderr,
    )

    for face_name, shape_name, color_name in tqdm(required_triplets, desc="Build v8 FS/Align cache", leave=False):
        face_path = find_image_path(ACTIVE_FACE_ROOT, face_name)
        shape_path = find_image_path(ACTIVE_SHAPE_ROOT, shape_name)
        color_path = find_image_path(ACTIVE_COLOR_ROOT, color_name)

        align_shape, align_color, name_to_embed = hair_fast(
            face_path,
            shape_path,
            color_path,
            align_flag=True,
        )

        if not fs_path("face", face_name).exists():
            save_latents(ACTIVE_DATASET_DIR, "FS", fs_cache_name("face", face_name), latent_in=name_to_embed["face"]["S"])
        if not fs_path("shape", shape_name).exists():
            save_latents(ACTIVE_DATASET_DIR, "FS", fs_cache_name("shape", shape_name), latent_in=name_to_embed["shape"]["S"])
        if not fs_path("color", color_name).exists():
            save_latents(ACTIVE_DATASET_DIR, "FS", fs_cache_name("color", color_name), latent_in=name_to_embed["color"]["S"])

        if USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, "shape", shape_name).exists()):
            save_latents(
                ACTIVE_DATASET_DIR,
                "Align",
                align_cache_name(face_name, "shape", shape_name),
                latent_F=align_shape["latent_F_align"],
            )
        if USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, "color", color_name).exists()):
            save_latents(
                ACTIVE_DATASET_DIR,
                "Align",
                align_cache_name(face_name, "color", color_name),
                latent_F=align_color["latent_F_align"],
            )
        if USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_has_target_hair(mask_path(face_name, "shape", shape_name))):
            save_latents(
                ACTIVE_DATASET_DIR,
                "Masks",
                align_cache_name(face_name, "shape", shape_name),
                remove_mask=build_remove_protect_mask(align_shape),
                target_hair=align_shape["HM_X"].float(),
            )
        if USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_has_target_hair(mask_path(face_name, "color", color_name))):
            save_latents(
                ACTIVE_DATASET_DIR,
                "Masks",
                align_cache_name(face_name, "color", color_name),
                remove_mask=build_remove_protect_mask(align_color),
                target_hair=align_color["HM_X"].float(),
            )

    del hair_fast
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_preview(path: Path, row_tensors: list[torch.Tensor]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for tensor in row_tensors:
        if tensor.dim() == 4:
            tensor = tensor[0]
        tiles.append(((tensor + 1) / 2).detach().cpu().clamp(0, 1))
    panel = torch.cat(tiles, dim=2)
    save_image(panel, path)


def mask_to_preview(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float().clamp(0, 1)
    if mask.size(1) == 1:
        mask = mask.repeat(1, 3, 1, 1)
    return mask * 2 - 1


PARSING_HAIR_LABEL = 10
PARSING_HAT_LABEL = 11
PARSING_FACE_PROTECT_LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12)
PARSING_NECK_PROTECT_LABELS = (13, 14)
PARSING_BODY_PROTECT_LABELS = (15,)
PARSING_SKIN_PROTECT_LABELS = PARSING_FACE_PROTECT_LABELS + PARSING_NECK_PROTECT_LABELS
PARSING_SUBJECT_PROTECT_LABELS = PARSING_SKIN_PROTECT_LABELS + PARSING_BODY_PROTECT_LABELS


def parsing_label_mask(parsing_mask: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(parsing_mask, dtype=torch.bool)
    for label in labels:
        mask |= parsing_mask == label
    return mask.float()


def dilate_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask.float().clamp(0, 1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return tnf.max_pool2d(mask.float(), kernel_size=2 * width + 1, stride=1, padding=width).clamp(0, 1)


class MaskPrepHelper:
    def __init__(self, device: torch.device):
        self.device = device
        self.dilate_erosion = DilateErosion(device=str(device))
        self.net = Net(
            Namespace(
                size=1024,
                ckpt="pretrained_models/StyleGAN/ffhq.pt",
                channel_multiplier=2,
                latent=512,
                n_mlp=8,
                device=str(device),
            )
        )
        self.seg = BiSeNet(n_classes=16).to(device).eval()
        self.seg.load_state_dict(torch.load("pretrained_models/BiSeNet/seg.pth", map_location=device))
        toggle_grad(self.seg, False)
        toggle_grad(self.net.generator, False)
        self.net.generator.eval()
        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)

    @torch.no_grad()
    def generate_mask(self, image: torch.Tensor, return_keep: bool = False):
        image_512 = (self.downsample_512((image + 1) / 2) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(image_512)
        current_mask = torch.argmax(down_seg, dim=1).long()
        hair_mask = torch.where(
            current_mask == PARSING_HAIR_LABEL,
            torch.ones_like(current_mask, dtype=torch.float32),
            torch.zeros_like(current_mask, dtype=torch.float32),
        )
        hair_mask = tnf.interpolate(hair_mask.unsqueeze(1), size=(256, 256), mode="nearest")
        hair_mask_dilate, hair_mask_erode = self.dilate_erosion.mask(hair_mask)
        if return_keep:
            non_hair_subject = (
                (current_mask > 0)
                & (current_mask != PARSING_HAIR_LABEL)
                & (current_mask != PARSING_HAT_LABEL)
            ).float()
            subject_guard = parsing_label_mask(current_mask, PARSING_SUBJECT_PROTECT_LABELS)
            skin_guard = parsing_label_mask(current_mask, PARSING_SKIN_PROTECT_LABELS)
            face_guard = parsing_label_mask(current_mask, PARSING_FACE_PROTECT_LABELS)
            neck_guard = parsing_label_mask(current_mask, PARSING_NECK_PROTECT_LABELS)

            non_hair_subject = tnf.interpolate(non_hair_subject.unsqueeze(1), size=(256, 256), mode="nearest")
            subject_guard = tnf.interpolate(subject_guard.unsqueeze(1), size=(256, 256), mode="nearest")
            skin_guard = tnf.interpolate(skin_guard.unsqueeze(1), size=(256, 256), mode="nearest")
            face_guard = tnf.interpolate(face_guard.unsqueeze(1), size=(256, 256), mode="nearest")
            neck_guard = tnf.interpolate(neck_guard.unsqueeze(1), size=(256, 256), mode="nearest")

            subject_guard = dilate_mask(subject_guard, 2)
            skin_guard = dilate_mask(skin_guard, 2)
            face_guard = dilate_mask(face_guard, 2)
            neck_guard = dilate_mask(neck_guard, 2)
            return (
                hair_mask_dilate,
                hair_mask_erode,
                non_hair_subject.clamp(0, 1),
                subject_guard.clamp(0, 1),
                skin_guard.clamp(0, 1),
                face_guard.clamp(0, 1),
                neck_guard.clamp(0, 1),
            )
        return hair_mask_dilate, hair_mask_erode


def prepare_item(exp, dataset_dir: Path, face_root: Path, color_root: Path):
    face_name, shape_name, color_name = exp

    try:
        color_s = torch.from_numpy(np.load(dataset_dir / "FS" / fs_cache_name("color", color_name))["latent_in"]).squeeze(0)
        align_s = torch.from_numpy(np.load(dataset_dir / "FS" / fs_cache_name("face", face_name))["latent_in"]).squeeze(0)
        align_f_shape = torch.from_numpy(
            np.load(dataset_dir / "Align" / align_cache_name(face_name, "shape", shape_name))["latent_F"]
        ).squeeze(0)
        with np.load(dataset_dir / "Masks" / align_cache_name(face_name, "shape", shape_name)) as mask_data:
            remove_mask_shape = torch.from_numpy(np.array(mask_data["remove_mask"])).squeeze(0)
            if "target_hair" in mask_data.files:
                target_hair_shape = torch.from_numpy(np.array(mask_data["target_hair"])).squeeze(0)
            else:
                target_hair_shape = torch.zeros_like(remove_mask_shape)

        align_f_color = torch.from_numpy(
            np.load(dataset_dir / "Align" / align_cache_name(face_name, "color", color_name))["latent_F"]
        ).squeeze(0)
        with np.load(dataset_dir / "Masks" / align_cache_name(face_name, "color", color_name)) as mask_data:
            remove_mask_color = torch.from_numpy(np.array(mask_data["remove_mask"])).squeeze(0)
            if "target_hair" in mask_data.files:
                target_hair_color = torch.from_numpy(np.array(mask_data["target_hair"])).squeeze(0)
            else:
                target_hair_color = torch.zeros_like(remove_mask_color)

        with Image.open(find_image_path(color_root, color_name)) as color_image:
            color_i = T.functional.normalize(T.functional.to_tensor(color_image.convert("RGB")), [0.5], [0.5])
        with Image.open(find_image_path(face_root, face_name)) as face_image:
            face_i = T.functional.normalize(T.functional.to_tensor(face_image.convert("RGB")), [0.5], [0.5])
        return (
            color_s,
            align_s,
            align_f_shape,
            remove_mask_shape,
            target_hair_shape,
            align_f_color,
            remove_mask_color,
            target_hair_color,
            color_i,
            face_i,
        )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return None


class BlendingDatasetV8(Dataset):
    def __init__(
        self,
        exps: list[tuple[str, str, str]],
        dataset_dir: Path,
        face_root: Path,
        color_root: Path,
    ):
        super().__init__()
        base_exps = [(p1, p2, p3) for (p1, p2, p3) in exps]
        if ACTIVE_SHAPE_ROOT.resolve() == ACTIVE_COLOR_ROOT.resolve():
            self.exps = base_exps + [(p1, p3, p2) for (p1, p2, p3) in exps]
        else:
            self.exps = base_exps
        self.dataset_dir = dataset_dir
        self.face_root = face_root
        self.color_root = color_root
        print(f"dataset pairs: {len(self.exps)}", file=sys.stderr)

    def __len__(self):
        return len(self.exps)

    def __getitem__(self, idx):
        item = prepare_item(self.exps[idx], self.dataset_dir, self.face_root, self.color_root)
        if item is None:
            raise RuntimeError(f"Failed to prepare blending item at index {idx}")
        return item


class BlendingTrainerV8:
    def __init__(
        self,
        model: BlendingModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        helper: MaskPrepHelper,
    ):
        self.device = helper.device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.helper = helper
        self.color_config = ColorConditionConfigV8(
            chroma_no_edit_threshold=USER_CHROMA_NO_EDIT_THRESHOLD_V8,
            chroma_full_edit_threshold=USER_CHROMA_FULL_EDIT_THRESHOLD_V8,
            lightness_no_edit_threshold=USER_LIGHTNESS_NO_EDIT_THRESHOLD_V8,
            lightness_full_edit_threshold=USER_LIGHTNESS_FULL_EDIT_THRESHOLD_V8,
            max_global_l_shift=USER_MAX_GLOBAL_L_SHIFT_V8,
            min_safe_fraction=USER_MIN_SAFE_REFERENCE_FRACTION_V8,
            highlight_mad_scale=USER_HIGHLIGHT_MAD_SCALE,
            highlight_global_min_margin=USER_HIGHLIGHT_GLOBAL_MIN_MARGIN,
            highlight_local_l_margin=USER_HIGHLIGHT_LOCAL_L_MARGIN,
            highlight_local_c_margin=USER_HIGHLIGHT_LOCAL_C_MARGIN,
            highlight_chroma_ratio=USER_HIGHLIGHT_CHROMA_RATIO,
        )
        self.grad_accum_steps = int(USER_GRAD_ACCUM_STEPS)
        self.best_loss = float("inf")
        self.output_ckpt_dir = ACTIVE_OUTPUT_DIR / "checkpoints"
        self.output_val_dir = ACTIVE_OUTPUT_DIR / "val_images"
        self.output_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)
        self.fid_calc = None
        if USER_USE_FID and Path(USER_FID_DATASET).exists():
            self.fid_calc = get_fid_calc(USER_FID_CACHE, str(USER_FID_DATASET), device=self.device)

    def prepare_batch(self, batch):
        (
            color_s,
            align_s,
            align_f_shape,
            remove_mask_shape,
            target_hair_shape,
            align_f_color,
            remove_mask_color,
            target_hair_color,
            color_i,
            face_i,
        ) = batch
        del align_f_color, remove_mask_color, target_hair_color
        align_f = align_f_shape
        remove_mask = remove_mask_shape
        target_hair = target_hair_shape
        color_s, align_s, align_f, remove_mask, target_hair, color_i, face_i = [
            item.to(self.device, non_blocking=True)
            for item in (color_s, align_s, align_f, remove_mask, target_hair, color_i, face_i)
        ]
        remove_mask = remove_mask.float().clamp(0, 1)
        if remove_mask.dim() == 3:
            remove_mask = remove_mask.unsqueeze(1)
        target_hair = target_hair.float().clamp(0, 1)
        if target_hair.dim() == 3:
            target_hair = target_hair.unsqueeze(1)

        with torch.no_grad():
            hm_3d, hm_3e = self.helper.generate_mask(color_i)
            (
                hm_1d,
                _,
                source_keep_mask,
                source_subject_guard,
                source_skin_guard,
                source_face_guard,
                source_neck_guard,
            ) = self.helper.generate_mask(
                face_i,
                return_keep=True,
            )
            i_x, _ = self.helper.net.generator(
                [align_s],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=align_f,
            )
            (
                hm_xd,
                hm_xe,
                face_keep_mask,
                target_subject_guard,
                target_skin_guard,
                target_face_guard,
                target_neck_guard,
            ) = self.helper.generate_mask(
                i_x,
                return_keep=True,
            )
            i_x_256 = self.helper.downsample_256(i_x)
            face_i_256 = self.helper.downsample_256(face_i)
            color_i_256 = self.helper.downsample_256(color_i)

        cached_hair_d, cached_hair_e = self.helper.dilate_erosion.mask(target_hair)
        has_cached_hair = target_hair.flatten(1).sum(dim=1) >= USER_SAFE_HAIR_MIN_PIXELS
        target_hair_d = torch.where(
            has_cached_hair.view(-1, 1, 1, 1),
            torch.maximum(hm_xd, cached_hair_d),
            hm_xd,
        ).clamp(0, 1)
        target_hair_e = torch.where(
            has_cached_hair.view(-1, 1, 1, 1),
            torch.maximum(hm_xe, cached_hair_e),
            hm_xe,
        ).clamp(0, 1)

        target_mask = (1 - hm_1d) * (1 - hm_3d) * (1 - target_hair_d)
        neck_hair_override = (USER_TARGET_HAIR_NECK_OVERRIDE * target_hair_e).clamp(0, 1)
        target_neck_visible = (target_neck_guard * (1.0 - neck_hair_override)).clamp(0, 1)
        source_neck_visible = (source_neck_guard * (1.0 - neck_hair_override)).clamp(0, 1)
        skin_color_block = (
            target_face_guard
            + target_neck_visible
            + 0.55 * source_face_guard
            + 0.55 * source_neck_visible
        ).clamp(0, 1)
        skin_protect_mask = (
            target_face_guard
            + target_neck_visible
            + 0.45 * source_face_guard
            + 0.45 * source_neck_visible
        ).clamp(0, 1)
        remove_color_block = (
            remove_mask * (1.0 - target_hair_d)
            + USER_REMOVE_BLOCK_IN_TARGET_HAIR * remove_mask * target_hair_d
        ).clamp(0, 1)
        color_transfer_eroded = (
            target_hair_e
            * (1.0 - remove_color_block)
            * (1.0 - USER_FACE_NECK_COLOR_BLOCK * skin_color_block)
        ).clamp(0, 1)
        color_transfer_core = (
            ((0.72 * target_hair_e) + (0.28 * target_hair_d))
            * (1.0 - remove_color_block)
            * (1.0 - USER_FACE_NECK_COLOR_BLOCK * skin_color_block)
        ).clamp(0, 1)
        color_transfer_fallback = (
            target_hair_d
            * (1.0 - 0.45 * remove_color_block)
            * (1.0 - USER_FACE_NECK_COLOR_BLOCK * skin_color_block)
        ).clamp(0, 1)
        needs_fallback = color_transfer_eroded.flatten(1).sum(dim=1) < USER_SAFE_HAIR_MIN_PIXELS
        color_transfer_mask = torch.where(
            needs_fallback.view(-1, 1, 1, 1),
            color_transfer_fallback,
            color_transfer_core,
        ).clamp(0, 1)
        subject_protect_mask = (
            (
                face_keep_mask
                + source_keep_mask
                + target_subject_guard
                + 0.60 * source_subject_guard
                + target_skin_guard
                + 0.50 * source_skin_guard
            )
            * (1.0 - color_transfer_mask)
        ).clamp(0, 1)
        skin_protect_mask = (skin_protect_mask * (1.0 - color_transfer_mask)).clamp(0, 1)
        satd_protect_mask = (
            remove_color_block
            + subject_protect_mask
            + skin_protect_mask
            + 0.35 * target_mask
            + 0.25 * (1.0 - target_hair_d) * (1.0 - hm_3e)
        ).clamp(0, 1)
        with torch.no_grad():
            condition_bundle = build_color_condition_bundle(
                reference_image=color_i_256,
                reference_hair_mask=hm_3e,
                base_image=i_x_256,
                target_hair_mask=color_transfer_mask,
                config=self.color_config,
            )
        color_reference_mask = condition_bundle["safe_ref_mask"]

        valid = color_reference_mask.flatten(1).any(dim=1) & color_transfer_mask.flatten(1).any(dim=1)
        if not valid.any():
            return None

        return {
            "color_s": color_s[valid],
            "align_s": align_s[valid],
            "align_f": align_f[valid],
            "color_i": color_i_256[valid],
            "face_i": face_i_256[valid],
            "base_i": i_x_256[valid],
            "target_mask": target_mask[valid],
            "satd_protect_mask": satd_protect_mask[valid],
            "remove_mask": remove_color_block[valid],
            "color_transfer_mask": color_transfer_mask[valid],
            "face_keep_mask": face_keep_mask[valid],
            "skin_protect_mask": skin_protect_mask[valid],
            "reference_hair_mask": hm_3e[valid],
            "safe_ref_mask": condition_bundle["safe_ref_mask"][valid],
            "rejected_highlight_mask": condition_bundle["rejected_highlight_mask"][valid],
            "color_descriptor": condition_bundle["descriptor"][valid],
            "chroma_need_gate": condition_bundle["chroma_need_gate"][valid],
            "lightness_need_gate": condition_bundle["lightness_need_gate"][valid],
            "edit_need_gate": condition_bundle["edit_need_gate"][valid],
            "pseudo_lab": condition_bundle["pseudo_lab"][valid],
            "pseudo_rgb": condition_bundle["pseudo_rgb"][valid],
            "color_proxy": condition_bundle["color_proxy"][valid],
            "condition_metrics": {
                key: value[valid]
                for key, value in condition_bundle["metrics"].items()
            },
        }

    @staticmethod
    def masked_l1(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        denom = mask.sum().clamp_min(1.0) * source.size(1)
        return (torch.abs(source - target) * mask).sum() / denom

    @staticmethod
    def masked_smooth_l1(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        element_loss = tnf.smooth_l1_loss(source, target, reduction="none")
        denominator = mask.sum().clamp_min(1.0) * source.size(1)
        return (element_loss * mask).sum() / denominator

    @staticmethod
    def masked_mean_value(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        return (value * mask).sum() / (mask.sum().clamp_min(1.0) * value.size(1))

    @staticmethod
    def masked_fraction_above(value: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        return (((value > threshold).float() * mask).sum() / mask.sum().clamp_min(1.0))

    @staticmethod
    def masked_q95(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per_sample = []
        for sample_value, sample_mask in zip(value.detach(), mask.detach()):
            selected = sample_value[sample_mask.expand_as(sample_value) > 0.05]
            per_sample.append(torch.quantile(selected, 0.95) if selected.numel() else value.new_zeros(()))
        return torch.stack(per_sample).mean()

    def calc_loss(
        self,
        i_gen: torch.Tensor,
        prepared: dict[str, object],
        encoder_aux: dict[str, torch.Tensor],
    ):
        i_face = prepared["face_i"]
        i_base = prepared["base_i"]
        mask_face = prepared["target_mask"]
        mask_gen_hair = prepared["color_transfer_mask"]
        satd_protect_mask = prepared["satd_protect_mask"]
        face_keep_mask = prepared["face_keep_mask"]
        skin_protect_mask = prepared["skin_protect_mask"]
        remove_mask = prepared["remove_mask"]
        pseudo_lab = prepared["pseudo_lab"]
        pseudo_rgb = prepared["pseudo_rgb"]

        mask_gen_hair = mask_gen_hair.float().clamp(0, 1)
        satd_protect_mask = satd_protect_mask.float().clamp(0, 1)
        face_keep_mask = face_keep_mask.float().clamp(0, 1)
        skin_protect_mask = skin_protect_mask.float().clamp(0, 1)
        remove_mask = remove_mask.float().clamp(0, 1)

        gen_face_embed = self.model.get_image_embed(i_gen * mask_face)
        face_embed = self.model.get_image_embed(i_face * mask_face)
        face_loss = (1 - tnf.cosine_similarity(gen_face_embed, face_embed)).mean()

        hair_loss = i_gen.sum() * 0.0

        gen_lab = rgb_to_lab(i_gen)
        base_lab = rgb_to_lab(i_base)
        gen_luma = gen_lab[:, 0:1]
        base_luma = base_lab[:, 0:1]
        gen_chroma = gen_lab[:, 1:3]
        base_chroma = base_lab[:, 1:3]

        pseudo_ab_loss = self.masked_smooth_l1(
            gen_chroma / 110.0,
            pseudo_lab[:, 1:3] / 110.0,
            mask_gen_hair,
        )
        pseudo_rgb_loss = self.masked_l1(i_gen, pseudo_rgb, mask_gen_hair)
        pseudo_luma_loss = self.masked_smooth_l1(
            gen_luma / 100.0,
            pseudo_lab[:, 0:1] / 100.0,
            mask_gen_hair,
        )
        luma_excess = gen_luma - pseudo_lab[:, 0:1]
        positive_luma = torch.relu(luma_excess - USER_LUMA_EXCESS_MARGIN)
        positive_luma_loss = self.masked_mean_value(positive_luma / 100.0, mask_gen_hair)
        hp_gen = gen_luma - gaussian_blur2d(gen_luma, radius=3)
        hp_base = base_luma - gaussian_blur2d(base_luma, radius=3)
        hf_luma_excess = torch.relu(
            hp_gen.abs() - hp_base.abs() - USER_HF_LUMA_EXCESS_MARGIN
        )
        hf_luma_loss = self.masked_mean_value(hf_luma_excess / 100.0, mask_gen_hair)

        face_keep_region = (face_keep_mask * satd_protect_mask).clamp(0, 1)
        protect_region = (face_keep_region + remove_mask).clamp(0, 1)
        face_keep_loss = self.masked_l1(i_gen, i_base, face_keep_region)
        remove_keep_loss = self.masked_l1(i_gen, i_base, remove_mask)
        protect_chroma_keep_loss = self.masked_l1(gen_chroma, base_chroma, protect_region)
        skin_chroma_keep_loss = self.masked_l1(gen_chroma, base_chroma, skin_protect_mask)
        skin_rgb_keep_loss = self.masked_l1(i_gen, i_base, skin_protect_mask)
        correction_norm_loss = (
            encoder_aux["correction_norm"]
            / encoder_aux["direct_delta_norm"].clamp_min(1.0)
        ).mean()

        total_loss = (
            USER_FACE_CLIP_LOSS_WEIGHT * face_loss
            + USER_PSEUDO_AB_LOSS_WEIGHT * pseudo_ab_loss
            + USER_PSEUDO_RGB_LOSS_WEIGHT * pseudo_rgb_loss
            + USER_PSEUDO_LUMA_LOSS_WEIGHT * pseudo_luma_loss
            + USER_POSITIVE_LUMA_EXCESS_WEIGHT * positive_luma_loss
            + USER_HF_LUMA_EXCESS_WEIGHT * hf_luma_loss
            + USER_FACE_KEEP_L1_LOSS_WEIGHT * face_keep_loss
            + USER_REMOVE_KEEP_L1_LOSS_WEIGHT * remove_keep_loss
            + USER_PROTECT_CHROMA_KEEP_LOSS_WEIGHT * protect_chroma_keep_loss
            + USER_SKIN_CHROMA_KEEP_LOSS_WEIGHT * skin_chroma_keep_loss
            + USER_SKIN_RGB_KEEP_LOSS_WEIGHT * skin_rgb_keep_loss
            + USER_CORRECTION_NORM_WEIGHT * correction_norm_loss
        )
        return total_loss, {
            "face_loss": face_loss,
            "hair_loss": hair_loss,
            "pseudo_ab": pseudo_ab_loss,
            "pseudo_rgb": pseudo_rgb_loss,
            "pseudo_luma": pseudo_luma_loss,
            "positive_luma": positive_luma_loss,
            "hf_luma": hf_luma_loss,
            "face_keep_l1": face_keep_loss,
            "remove_keep_l1": remove_keep_loss,
            "protect_chroma_keep": protect_chroma_keep_loss,
            "skin_chroma_keep": skin_chroma_keep_loss,
            "skin_rgb_keep": skin_rgb_keep_loss,
            "correction_norm": correction_norm_loss,
            "result_to_pseudo_ab_l2": self.masked_mean_value(
                torch.linalg.vector_norm(gen_chroma - pseudo_lab[:, 1:3], dim=1, keepdim=True),
                mask_gen_hair,
            ),
            "result_hue_error": self.masked_mean_value(
                torch.rad2deg(
                    torch.acos(
                        (
                            (gen_chroma * pseudo_lab[:, 1:3]).sum(dim=1, keepdim=True)
                            / (
                                torch.linalg.vector_norm(gen_chroma, dim=1, keepdim=True)
                                * torch.linalg.vector_norm(pseudo_lab[:, 1:3], dim=1, keepdim=True)
                            ).clamp_min(1e-4)
                        ).clamp(-1, 1)
                    )
                ),
                mask_gen_hair,
            ),
            "mean_l_excess": self.masked_mean_value(torch.relu(luma_excess), mask_gen_hair),
            "q95_l_excess": self.masked_q95(luma_excess, mask_gen_hair),
            "frac_l_excess_gt8": self.masked_fraction_above(luma_excess, mask_gen_hair, 8.0),
            "frac_l_excess_gt12": self.masked_fraction_above(luma_excess, mask_gen_hair, 12.0),
            "frac_l_excess_gt16": self.masked_fraction_above(luma_excess, mask_gen_hair, 16.0),
            "loss": total_loss,
        }

    def save_checkpoint(self, epoch: int, best_loss: float, name: str):
        model_state_dict = self.model.state_dict()
        saved_state_dict = {key: value for key, value in model_state_dict.items() if not key.startswith("clip_model.")}
        torch.save(
            {
                "arch": DIRECT_COLOR_ARCH_V8_2,
                "epoch": epoch,
                "best_loss": best_loss,
                "clip": USER_CLIP_MODEL,
                "adapter_config": {
                    "direct_mix_init": USER_DIRECT_MIX_INIT,
                    "direct_mix_floor": USER_DIRECT_MIX_FLOOR,
                    "correction_budget_ratio": USER_CORRECTION_BUDGET_RATIO,
                },
                "model_state_dict": saved_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.output_ckpt_dir / f"{name}.pth",
        )

    def load_resume_checkpoint(self) -> int:
        if not USER_RESUME_CHECKPOINT:
            return 0

        resume_path = Path(USER_RESUME_CHECKPOINT)
        if not resume_path.exists():
            raise FileNotFoundError(f"Cannot find USER_RESUME_CHECKPOINT: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=self.device)
        checkpoint_arch = checkpoint.get("arch") if isinstance(checkpoint, dict) else None
        if checkpoint_arch != DIRECT_COLOR_ARCH_V8_2:
            raise RuntimeError(
                f"Refusing to resume incompatible BlendingV8 checkpoint {resume_path}: "
                f"arch={checkpoint_arch!r}, required={DIRECT_COLOR_ARCH_V8_2!r}"
            )
        adapter_config = checkpoint.get("adapter_config", {})
        expected_adapter_config = {
            "direct_mix_init": self.model.direct_mix_init,
            "direct_mix_floor": self.model.direct_mix_floor,
            "correction_budget_ratio": self.model.correction_budget_ratio,
        }
        mismatched_config = {
            key: (adapter_config[key], expected_value)
            for key, expected_value in expected_adapter_config.items()
            if key in adapter_config and float(adapter_config[key]) != float(expected_value)
        }
        if mismatched_config:
            raise RuntimeError(
                f"Resume adapter_config does not match current V2 config: {mismatched_config}"
            )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        report = load_direct_color_adapter_state_v8(self.model, state_dict)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_loss = float(checkpoint.get("best_loss", self.best_loss))
        start_epoch = int(checkpoint.get("epoch", 0))
        print(
            f"[blending_v8] resumed from {resume_path} "
            f"with {len(report['loaded'])} strict V2 adapter tensors; "
            f"start_epoch={start_epoch + 1} best_loss={self.best_loss:.6f}",
            file=sys.stderr,
        )
        return start_epoch

    @staticmethod
    def build_generator_latent(align_s: torch.Tensor, blend_s: torch.Tensor) -> torch.Tensor:
        # This matches the author's training code. With start_layer=4 the
        # prefix is not rendered, but keeping the contract explicit avoids
        # accidental dependence if the generator call changes later.
        if USER_AUTHOR_ZERO_PREFIX_TRAIN:
            prefix = torch.zeros_like(align_s[:, :6])
        else:
            prefix = align_s[:, :6]
        return torch.cat((prefix, blend_s), dim=1)

    def configure_stage(self, epoch: int) -> tuple[str, bool]:
        stage_a = epoch < USER_STAGE_A_EPOCHS
        correction_enabled = not stage_a
        learning_rate = USER_LR_STAGE_A if stage_a else USER_LR_STAGE_B
        self.model.set_correction_trainable(correction_enabled)
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        return ("A" if stage_a else "B"), correction_enabled

    def run_adapter(
        self,
        prepared: dict[str, object],
        *,
        correction_enabled: bool,
        layer_mix_override: float | None = None,
    ):
        return self.model(
            latent_face=prepared["align_s"][:, 6:],
            latent_color=prepared["color_s"][:, 6:],
            color_descriptor=prepared["color_descriptor"],
            chroma_need_gate=prepared["chroma_need_gate"],
            lightness_need_gate=prepared["lightness_need_gate"],
            edit_need_gate=prepared["edit_need_gate"],
            correction_enabled=correction_enabled,
            layer_mix_override=layer_mix_override,
            return_aux=True,
        )

    def render_blend_tail(self, prepared: dict[str, object], blend_s: torch.Tensor) -> torch.Tensor:
        latent_in = self.build_generator_latent(prepared["align_s"], blend_s)
        image, _ = self.helper.net.generator(
            [latent_in],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=prepared["align_f"],
        )
        return self.helper.downsample_256(image)

    def train_one_epoch(self, epoch: int):
        stage, correction_enabled = self.configure_stage(epoch)
        self.model.train()
        self.model.clip_model.eval()
        running_loss = 0.0
        running_steps = 0
        accumulated_batches = 0
        last_grad_norm = 0.0
        running_metrics: dict[str, float] = {}
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(self.train_loader, desc=f"Blend train {epoch + 1}/{USER_EPOCHS}", leave=False)
        for batch in progress:
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue

            blend_s, encoder_aux = self.run_adapter(
                prepared,
                correction_enabled=correction_enabled,
            )
            latent_in = self.build_generator_latent(prepared["align_s"], blend_s)
            i_g, _ = self.helper.net.generator(
                [latent_in],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=prepared["align_f"],
            )
            loss, loss_info = self.calc_loss(self.helper.downsample_256(i_g), prepared, encoder_aux)

            (loss / self.grad_accum_steps).backward()
            accumulated_batches += 1
            if accumulated_batches == self.grad_accum_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                last_grad_norm = float(grad_norm)

            running_loss += float(loss.item())
            running_steps += 1
            batch_metrics = {
                "loss_pseudo_ab": loss_info["pseudo_ab"],
                "loss_pseudo_rgb": loss_info["pseudo_rgb"],
                "loss_pseudo_luma": loss_info["pseudo_luma"],
                "loss_positive_luma": loss_info["positive_luma"],
                "loss_hf_luma": loss_info["hf_luma"],
                "loss_correction_norm": loss_info["correction_norm"],
                "mean_chroma_need_gate": prepared["chroma_need_gate"].mean(),
                "mean_lightness_need_gate": prepared["lightness_need_gate"].mean(),
                "mean_edit_need_gate": prepared["edit_need_gate"].mean(),
                "mean_safe_fraction": prepared["condition_metrics"]["safe_fraction"].mean(),
                "mean_direct_component_norm": encoder_aux["direct_component_norm"].mean(),
                "mean_correction_norm": encoder_aux["correction_norm"].mean(),
                "mean_layer_mix": encoder_aux["layer_mix_mean"].mean(),
            }
            for key, value in batch_metrics.items():
                running_metrics[key] = running_metrics.get(key, 0.0) + float(value.item())
            progress.set_postfix(
                loss=float(loss.item()),
                ab=float(loss_info["pseudo_ab"].item()),
                rgb=float(loss_info["pseudo_rgb"].item()),
                luma=float(loss_info["pseudo_luma"].item()),
                excess=float(loss_info["positive_luma"].item()),
                gate=float(prepared["edit_need_gate"].mean().item()),
                skin=float(loss_info["skin_chroma_keep"].item()),
                grad=last_grad_norm,
                accum=f"{accumulated_batches}/{self.grad_accum_steps}",
            )

        if accumulated_batches:
            scale = self.grad_accum_steps / accumulated_batches
            if scale != 1.0:
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        averaged_metrics = {
            key: value / max(running_steps, 1)
            for key, value in running_metrics.items()
        }
        print(
            f"[blending_v8] epoch={epoch + 1} stage={stage} "
            f"lr={self.optimizer.param_groups[0]['lr']:.2e} correction={correction_enabled} train_components "
            + " ".join(f"{key}={value:.6f}" for key, value in averaged_metrics.items())
        )
        return running_loss / max(running_steps, 1)

    @torch.no_grad()
    def validate(
        self,
        epoch: int,
        *,
        correction_enabled: bool | None = None,
        output_dir_name: str | None = None,
    ):
        if correction_enabled is None:
            correction_enabled = epoch >= USER_STAGE_A_EPOCHS
        validation_label = "pretrain" if epoch < 0 else f"{epoch + 1}/{USER_EPOCHS}"
        self.model.eval()
        total_losses: dict[str, float] = {}
        total_diagnostics: dict[str, float] = {}
        total_steps = 0
        images_to_fid = []
        preview_rows = []
        direct_preview_rows = []
        validation_records = []

        for batch in tqdm(self.val_loader, desc=f"Blend val {validation_label}", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue

            bsz = prepared["color_s"].size(0)
            blend_s, encoder_aux = self.run_adapter(
                prepared,
                correction_enabled=correction_enabled,
            )
            latent_in = self.build_generator_latent(prepared["align_s"], blend_s)
            i_g, _ = self.helper.net.generator(
                [latent_in],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=prepared["align_f"],
            )
            i_g_256 = self.helper.downsample_256(i_g)
            loss, loss_info = self.calc_loss(i_g_256, prepared, encoder_aux)

            for key, value in loss_info.items():
                total_losses[key] = total_losses.get(key, 0.0) + float(value.item())
            batch_diagnostics = {
                "mean_chroma_need_gate": prepared["chroma_need_gate"].mean(),
                "mean_lightness_need_gate": prepared["lightness_need_gate"].mean(),
                "mean_edit_need_gate": prepared["edit_need_gate"].mean(),
                "mean_safe_fraction": prepared["condition_metrics"]["safe_fraction"].mean(),
                "mean_rejected_fraction": prepared["condition_metrics"]["rejected_fraction"].mean(),
                "mean_direct_delta_norm": encoder_aux["direct_delta_norm"].mean(),
                "mean_direct_component_norm": encoder_aux["direct_component_norm"].mean(),
                "mean_correction_norm": encoder_aux["correction_norm"].mean(),
                "mean_layer_mix": encoder_aux["layer_mix_mean"].mean(),
            }
            for key, value in batch_diagnostics.items():
                total_diagnostics[key] = total_diagnostics.get(key, 0.0) + float(value.item())
            total_steps += 1

            if self.fid_calc is not None:
                images_to_fid.append(T.Resize((299, 299))(((i_g + 1) / 2).clamp(0, 1)))

            generated_lab = rgb_to_lab(i_g_256)
            luma_excess = generated_lab[:, 0:1] - prepared["pseudo_lab"][:, 0:1]
            if len(preview_rows) < USER_LOG_IMAGE_COUNT:
                full_direct_s, _ = self.run_adapter(
                    prepared,
                    correction_enabled=False,
                    layer_mix_override=1.0,
                )
                initial_direct_s, _ = self.run_adapter(
                    prepared,
                    correction_enabled=False,
                    layer_mix_override=USER_DIRECT_MIX_INIT,
                )
                full_direct_i = self.render_blend_tail(prepared, full_direct_s)
                initial_direct_i = self.render_blend_tail(prepared, initial_direct_s)
                for idx in range(bsz):
                    excess_preview = (
                        (torch.relu(luma_excess[idx : idx + 1]) / 20.0).clamp(0, 1) * 2.0 - 1.0
                    ).repeat(1, 3, 1, 1)
                    preview_rows.append([
                        prepared["face_i"][idx : idx + 1],
                        prepared["color_i"][idx : idx + 1],
                        prepared["base_i"][idx : idx + 1],
                        i_g_256[idx : idx + 1],
                        mask_to_preview(prepared["color_transfer_mask"][idx : idx + 1]),
                        mask_to_preview(prepared["reference_hair_mask"][idx : idx + 1]),
                        mask_to_preview(prepared["safe_ref_mask"][idx : idx + 1]),
                        mask_to_preview(prepared["rejected_highlight_mask"][idx : idx + 1]),
                        prepared["pseudo_rgb"][idx : idx + 1],
                        excess_preview,
                    ])
                    direct_preview_rows.append([
                        prepared["base_i"][idx : idx + 1],
                        full_direct_i[idx : idx + 1],
                        initial_direct_i[idx : idx + 1],
                        i_g_256[idx : idx + 1],
                    ])
                    if len(preview_rows) >= USER_LOG_IMAGE_COUNT:
                        break

            for idx in range(bsz):
                sample_mask = prepared["color_transfer_mask"][idx : idx + 1]
                sample_gen_ab = generated_lab[idx : idx + 1, 1:3]
                sample_pseudo_ab = prepared["pseudo_lab"][idx : idx + 1, 1:3]
                hue_cosine = (
                    (sample_gen_ab * sample_pseudo_ab).sum(dim=1, keepdim=True)
                    / (
                        torch.linalg.vector_norm(sample_gen_ab, dim=1, keepdim=True)
                        * torch.linalg.vector_norm(sample_pseudo_ab, dim=1, keepdim=True)
                    ).clamp_min(1e-4)
                ).clamp(-1, 1)
                sample_excess = luma_excess[idx : idx + 1]
                validation_records.append({
                    "sample_index": len(validation_records),
                    "ref_base_ab_distance": float(
                        prepared["condition_metrics"]["ref_base_ab_distance"][idx].item()
                    ),
                    "delta_L_global": float(prepared["condition_metrics"]["delta_l_global"][idx].item()),
                    "chroma_need_gate": float(prepared["chroma_need_gate"][idx].item()),
                    "lightness_need_gate": float(prepared["lightness_need_gate"][idx].item()),
                    "edit_need_gate": float(prepared["edit_need_gate"][idx].item()),
                    "safe_fraction": float(prepared["condition_metrics"]["safe_fraction"][idx].item()),
                    "rejected_fraction": float(
                        prepared["condition_metrics"]["rejected_fraction"][idx].item()
                    ),
                    "result_to_pseudo_ab_l2": float(self.masked_mean_value(
                        torch.linalg.vector_norm(sample_gen_ab - sample_pseudo_ab, dim=1, keepdim=True),
                        sample_mask,
                    ).item()),
                    "result_hue_error": float(self.masked_mean_value(
                        torch.rad2deg(torch.acos(hue_cosine)), sample_mask
                    ).item()),
                    "mean_L_excess": float(self.masked_mean_value(torch.relu(sample_excess), sample_mask).item()),
                    "q95_L_excess": float(self.masked_q95(sample_excess, sample_mask).item()),
                    "frac_L_excess_gt8": float(self.masked_fraction_above(sample_excess, sample_mask, 8.0).item()),
                    "frac_L_excess_gt12": float(self.masked_fraction_above(sample_excess, sample_mask, 12.0).item()),
                    "frac_L_excess_gt16": float(self.masked_fraction_above(sample_excess, sample_mask, 16.0).item()),
                    "direct_delta_norm": float(encoder_aux["direct_delta_norm"][idx].item()),
                    "direct_component_norm": float(encoder_aux["direct_component_norm"][idx].item()),
                    "direct_mix_fraction": float(encoder_aux["direct_mix_fraction"][idx].item()),
                    "layer_mix_mean": float(encoder_aux["layer_mix_mean"][idx].item()),
                    "layer_mix_min": float(encoder_aux["layer_mix_min"][idx].item()),
                    "layer_mix_max": float(encoder_aux["layer_mix_max"][idx].item()),
                    "correction_raw_norm": float(encoder_aux["correction_raw_norm"][idx].item()),
                    "correction_norm": float(encoder_aux["correction_norm"][idx].item()),
                    "correction_budget": float(encoder_aux["correction_budget"][idx].item()),
                    "correction_budget_scale": float(encoder_aux["correction_budget_scale"][idx].item()),
                    "correction_to_direct_ratio": float(
                        encoder_aux["correction_to_direct_ratio"][idx].item()
                    ),
                    "total_delta_norm": float(encoder_aux["total_delta_norm"][idx].item()),
                    "total_to_direct_ratio": float(encoder_aux["total_to_direct_ratio"][idx].item()),
                })

        avg_losses = {key: value / max(total_steps, 1) for key, value in total_losses.items()}
        avg_diagnostics = {key: value / max(total_steps, 1) for key, value in total_diagnostics.items()}
        if self.fid_calc is not None and images_to_fid:
            avg_losses["fid_clip"] = float(self.fid_calc(torch.cat(images_to_fid)).item())

        epoch_dir = self.output_val_dir / (
            output_dir_name if output_dir_name is not None else f"epoch_{epoch + 1:03d}"
        )
        epoch_dir.mkdir(parents=True, exist_ok=True)
        with open(epoch_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(validation_records, handle, ensure_ascii=False, indent=2)
        if epoch < 0 or epoch % USER_SAVE_PREVIEW_EVERY == 0:
            for idx, row in enumerate(preview_rows):
                save_preview(epoch_dir / f"sample_{idx:03d}.png", row)
                save_preview(epoch_dir / f"sample_{idx:03d}_direct_diagnostic.png", direct_preview_rows[idx])

        print(
            f"[blending_v8] validation={validation_label} correction={correction_enabled} "
            f"val_loss={avg_losses['loss']:.6f} "
            f"val_face={avg_losses['face_loss']:.6f} "
            f"val_hair={avg_losses['hair_loss']:.6f} "
            f"loss_pseudo_ab={avg_losses['pseudo_ab']:.6f} "
            f"loss_pseudo_rgb={avg_losses['pseudo_rgb']:.6f} "
            f"loss_pseudo_luma={avg_losses['pseudo_luma']:.6f} "
            f"loss_positive_luma={avg_losses['positive_luma']:.6f} "
            f"loss_hf_luma={avg_losses['hf_luma']:.6f} "
            f"mean_edit_need_gate={avg_diagnostics['mean_edit_need_gate']:.6f} "
            f"mean_safe_fraction={avg_diagnostics['mean_safe_fraction']:.6f} "
            f"mean_direct_component_norm={avg_diagnostics['mean_direct_component_norm']:.6f} "
            f"mean_correction_norm={avg_diagnostics['mean_correction_norm']:.6f} "
            f"val_ab_error={avg_losses['result_to_pseudo_ab_l2']:.6f} "
            f"val_frac_excess_gt12={avg_losses['frac_l_excess_gt12']:.6f} "
            f"val_skin={avg_losses['skin_chroma_keep']:.6f}"
        )
        return avg_losses["loss"]

    def train_loop(self):
        start_epoch = self.load_resume_checkpoint()
        if start_epoch >= USER_EPOCHS:
            print(
                f"[blending_v8] resume checkpoint is already at epoch {start_epoch}; "
                f"USER_EPOCHS={USER_EPOCHS}, nothing to train.",
                file=sys.stderr,
            )
            return

        if start_epoch == 0:
            self.configure_stage(0)
            self.validate(
                -1,
                correction_enabled=False,
                output_dir_name="epoch_000_pretrain",
            )

        for epoch in range(start_epoch, USER_EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(
                epoch,
                correction_enabled=epoch >= USER_STAGE_A_EPOCHS,
            )
            print(f"[blending_v8] epoch={epoch + 1} train_loss={train_loss:.6f}")

            is_best = val_loss <= self.best_loss
            if is_best:
                self.best_loss = val_loss
            if (epoch + 1) % USER_SAVE_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(epoch + 1, self.best_loss, "last")
            if is_best:
                self.save_checkpoint(epoch + 1, self.best_loss, "best")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    atexit.register(clean_zombies)
    set_seed(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    triplets = read_triplets(ACTIVE_DATASET_DIR)
    if not triplets:
        raise RuntimeError(f"No 3-column experiments found in {ACTIVE_DATASET_DIR / 'dataset.exps'}")
    if len(triplets) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"dataset.exps is smaller than the validation split size ({ACTIVE_VAL_SIZE}) "
            f"for profile {USER_DATASET_PROFILE!r}."
        )

    ensure_dataset_cache_v8(triplets)
    train_exps, val_exps = train_test_split(triplets, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    helper = MaskPrepHelper(device)

    train_dataset = BlendingDatasetV8(train_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_COLOR_ROOT)
    val_dataset = BlendingDatasetV8(val_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_COLOR_ROOT)
    train_loader = DataLoader(
        train_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
    )

    model = BlendingModel(
        USER_CLIP_MODEL,
        direct_mix_init=USER_DIRECT_MIX_INIT,
        direct_mix_floor=USER_DIRECT_MIX_FLOOR,
        correction_budget_ratio=USER_CORRECTION_BUDGET_RATIO,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=USER_LR_STAGE_A,
        weight_decay=USER_WEIGHT_DECAY,
    )

    trainer = BlendingTrainerV8(model, optimizer, train_loader, val_loader, helper)
    print(
        f"[blending_v8] train_on_shape_satd_align=True "
        f"author_color_align_aux={USER_AUTHOR_COLOR_ALIGN_BATCH_PROB > 0} use_satd_v8={USER_USE_SATD_V8} "
        f"arch={DIRECT_COLOR_ARCH_V8_2} fresh_adapter=True "
        f"satd_checkpoint={USER_SATD_CHECKPOINT_V8} "
        f"batch_size={USER_BATCH_SIZE} grad_accum_steps={USER_GRAD_ACCUM_STEPS} "
        f"effective_batch_size={USER_BATCH_SIZE * USER_GRAD_ACCUM_STEPS} "
        f"pseudo_ab_w={USER_PSEUDO_AB_LOSS_WEIGHT} "
        f"pseudo_rgb_w={USER_PSEUDO_RGB_LOSS_WEIGHT} "
        f"pseudo_luma_w={USER_PSEUDO_LUMA_LOSS_WEIGHT} "
        f"positive_luma_w={USER_POSITIVE_LUMA_EXCESS_WEIGHT} "
        f"hf_luma_w={USER_HF_LUMA_EXCESS_WEIGHT} "
        f"correction_norm_w={USER_CORRECTION_NORM_WEIGHT} "
        f"chroma_gate_thresholds=({USER_CHROMA_NO_EDIT_THRESHOLD_V8},{USER_CHROMA_FULL_EDIT_THRESHOLD_V8}) "
        f"lightness_gate_thresholds=({USER_LIGHTNESS_NO_EDIT_THRESHOLD_V8},{USER_LIGHTNESS_FULL_EDIT_THRESHOLD_V8}) "
        f"max_global_l_shift={USER_MAX_GLOBAL_L_SHIFT_V8} "
        f"min_safe_reference_fraction={USER_MIN_SAFE_REFERENCE_FRACTION_V8} "
        f"direct_mix_init={USER_DIRECT_MIX_INIT} "
        f"direct_mix_floor={USER_DIRECT_MIX_FLOOR} "
        f"correction_budget_ratio={USER_CORRECTION_BUDGET_RATIO} "
        f"stage_a_epochs={USER_STAGE_A_EPOCHS} "
        f"lr_stage_a={USER_LR_STAGE_A} lr_stage_b={USER_LR_STAGE_B} "
        f"remove_keep_w={USER_REMOVE_KEEP_L1_LOSS_WEIGHT} "
        f"protect_chroma_keep_w={USER_PROTECT_CHROMA_KEEP_LOSS_WEIGHT} "
        f"skin_chroma_keep_w={USER_SKIN_CHROMA_KEEP_LOSS_WEIGHT} "
        f"skin_rgb_keep_w={USER_SKIN_RGB_KEEP_LOSS_WEIGHT} "
        f"remove_block_in_target_hair={USER_REMOVE_BLOCK_IN_TARGET_HAIR} "
        f"face_neck_color_block={USER_FACE_NECK_COLOR_BLOCK} "
        f"target_hair_neck_override={USER_TARGET_HAIR_NECK_OVERRIDE} "
        f"author_color_align_batch_prob={USER_AUTHOR_COLOR_ALIGN_BATCH_PROB} "
        f"author_zero_prefix_train={USER_AUTHOR_ZERO_PREFIX_TRAIN} "
        f"resume_checkpoint={USER_RESUME_CHECKPOINT or '<none>'}",
        file=sys.stderr,
    )
    trainer.train_loop()


if __name__ == "__main__":
    main()
