"""Shared immutable inputs and compositing helpers for V2.45 death tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F


def sample_token(sample_id: str) -> str:
    return hashlib.sha1(str(sample_id).encode("utf-8")).hexdigest()[:16]


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def manifest_sha256(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def composite(base_rgb: torch.Tensor, hair_rgb: torch.Tensor,
              alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float().clamp(0, 1)
    if alpha.shape[1] != 1:
        alpha = alpha[:, :1]
    return (alpha * hair_rgb.float() + (1.0 - alpha) * base_rgb.float()).clamp(0, 1)


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 0)
    if radius == 0:
        return mask.float().clamp(0, 1)
    return (-F.max_pool2d(-mask.float(), 2 * radius + 1, stride=1, padding=radius)).clamp(0, 1)


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 0)
    if radius == 0:
        return mask.float().clamp(0, 1)
    return F.max_pool2d(mask.float(), 2 * radius + 1, stride=1, padding=radius).clamp(0, 1)


def smoothstep(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    x = ((value.float() - low) / max(high - low, 1e-6)).clamp(0, 1)
    return x * x * (3.0 - 2.0 * x)


def save_cache(cache_dir: Path, sample_id: str, payload: Mapping[str, torch.Tensor]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{sample_token(sample_id)}.pt"
    torch.save({key: value.detach().cpu() for key, value in payload.items()}, path)
    return path


def load_cache(cache_dir: Path, sample_id: str) -> dict[str, torch.Tensor]:
    path = cache_dir / f"{sample_token(sample_id)}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing V2.45 common cache for {sample_id}: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 1.13
        return torch.load(path, map_location="cpu")


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": "v2.45.1", "count": len(rows), "manifest_sha256": manifest_sha256(rows), "samples": rows}, indent=2, ensure_ascii=True), encoding="utf-8")


def read_manifest(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in ("v2.45", "v2.45.1"):
        raise RuntimeError(f"Unsupported death-test manifest version: {payload.get('version')!r}")
    return list(payload.get("samples", []))


__all__ = ["composite", "dilate", "erode", "load_cache", "manifest_sha256", "read_manifest", "sample_token", "save_cache", "smoothstep", "tensor_sha256", "write_manifest"]
