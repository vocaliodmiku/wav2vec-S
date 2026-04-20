"""Causality unit test for CausalConformerBlock.

The contract: perturbing input at frame t' must not change the output at any
frame t < t'. This is the highest-value cheap check for the pre-flight build.
"""
from __future__ import annotations

import torch

from hpc_speech.model.conformer import CausalConformerBlock


def test_causal_conformer_block_is_frame_causal():
    torch.manual_seed(0)
    d = 64
    T = 32
    block = CausalConformerBlock(d=d, num_heads=4, ffn_mult=2, conv_kernel=5, dropout=0.0).eval()

    x = torch.randn(1, T, d)
    y = block(x)

    # Perturb a single future frame and assert all earlier outputs are unchanged.
    t_perturb = 20
    x2 = x.clone()
    x2[:, t_perturb:, :] += 10.0 * torch.randn_like(x2[:, t_perturb:, :])
    y2 = block(x2)

    # Outputs at frames [0, t_perturb) must be identical (up to fp noise).
    max_abs = (y[:, :t_perturb] - y2[:, :t_perturb]).abs().max().item()
    assert max_abs < 1e-5, f"causality violated: max abs delta at past frames = {max_abs}"


def test_causal_conformer_stack_is_frame_causal():
    # Two stacked blocks must still be causal.
    torch.manual_seed(1)
    d = 64
    T = 40
    blocks = torch.nn.Sequential(
        CausalConformerBlock(d=d, num_heads=4, ffn_mult=2, conv_kernel=7, dropout=0.0),
        CausalConformerBlock(d=d, num_heads=4, ffn_mult=2, conv_kernel=7, dropout=0.0),
    ).eval()

    x = torch.randn(2, T, d)
    y = blocks(x)

    t_perturb = 25
    x2 = x.clone()
    x2[:, t_perturb:, :] = torch.randn_like(x2[:, t_perturb:, :])
    y2 = blocks(x2)

    max_abs = (y[:, :t_perturb] - y2[:, :t_perturb]).abs().max().item()
    assert max_abs < 1e-5, f"causality violated in stack: max abs delta = {max_abs}"


if __name__ == "__main__":
    test_causal_conformer_block_is_frame_causal()
    test_causal_conformer_stack_is_frame_causal()
    print("OK: causality tests passed")
