"""V2.41/V2.42 runtime input contract."""

from __future__ import annotations

from models.v838_runtime_inputs import build_v838_runtime_inputs


def build_v841_runtime_inputs(**kwargs):
    return build_v838_runtime_inputs(**kwargs)


__all__ = ["build_v841_runtime_inputs"]
