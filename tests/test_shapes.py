"""Shape-level smoke tests for the pre-flight model."""
from __future__ import annotations

import torch

from hpc_speech.config import PreflightConfig, TrunkConfig
from hpc_speech.model.conformer import CausalConformerBlock
from hpc_speech.model.predictors import StateSummary, TopDownPredictor
from hpc_speech.model.trunk import HierarchicalTrunk


def test_conformer_shape():
    blk = CausalConformerBlock(d=64, num_heads=4, ffn_mult=2, conv_kernel=5).eval()
    x = torch.randn(2, 30, 64)
    assert blk(x).shape == (2, 30, 64)


def test_predictor_shape_and_lag():
    g = TopDownPredictor(d=32, rank=8, lag_k=4).eval()
    h = torch.randn(1, 20, 32)
    y = g(h)
    assert y.shape == h.shape
    # First k frames of y should equal U(tanh(V(0))) = U(tanh(bias_V)) broadcast.
    # Simpler check: y[:, 0] is independent of h[:, 4:].
    h2 = h.clone()
    h2[:, 4:] += 5.0
    y2 = g(h2)
    assert torch.allclose(y[:, :4], y2[:, :4], atol=1e-5)


def test_state_summary_shape():
    s = StateSummary(d=32, rank=8)
    assert s(torch.randn(2, 10, 32)).shape == (2, 10, 8)


def test_trunk_forward_shapes():
    cfg = TrunkConfig(num_levels=5, d=64, num_heads=4, ffn_mult=2, conv_kernel=5,
                      blocks_per_level=1, dropout=0.0, predictor_rank=16,
                      predictor_lag_k=2, state_rank=8,
                      tap_injection=(0, 1, 2, 3, -1))
    tap_keys = ["tap6", "tap12", "tap18", "tap24", ""]
    trunk = HierarchicalTrunk(cfg, frontend_dim=128, tap_keys=tap_keys).eval()
    B, T = 2, 20
    taps = {f"tap{lyr}": torch.randn(B, T, 128) for lyr in (6, 12, 18, 24)}
    out = trunk(taps)
    assert len(out["h"]) == 5 and all(h.shape == (B, T, 64) for h in out["h"])
    assert len(out["e"]) == 4 and all(e.shape == (B, T, 64) for e in out["e"])
    assert len(out["s"]) == 4 and all(s.shape == (B, T, 8) for s in out["s"])
    assert len(out["y_hat"]) == 4


if __name__ == "__main__":
    test_conformer_shape()
    test_predictor_shape_and_lag()
    test_state_summary_shape()
    test_trunk_forward_shapes()
    print("OK")
