"""V2.39 runtime contract; same tensors as V2.38 with explicit names."""

from __future__ import annotations

from models.v838_runtime_inputs import build_v838_runtime_inputs


def build_v839_runtime_inputs(**kwargs):
    return build_v838_runtime_inputs(**kwargs)


__all__ = ["build_v839_runtime_inputs"]
