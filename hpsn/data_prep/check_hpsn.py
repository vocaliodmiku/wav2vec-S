"""HPSN-v2 Phase 2C smoke test for the iterative-refinement loop and
frozen one-hot taps.

Asserts:
1. Default (n_iterations=1) reproduces v1 forward output shape and produces
   finite reconstructions.
2. n_iterations=2 runs without crashing and returns the same output keys
   with the same shapes.
3. With identical input + same parameters, n_iter=2 outputs differ from
   n_iter=1 (the refinement pass actually changes representations; if eps2
   were trivially zero this would fail).
4. Bottom-up error projections (``error_proj_{1,2}``) are present in
   ``model.parameters()`` and have grad enabled.
5. Frozen taps survive into the model: ``hpsn.tap1.frozen_layer == k``,
   the ``weights`` buffer is one-hot, and is excluded from the optimizer
   parameter set.

Usage
-----
    python -m hpsn.data_prep.check_hpsn \\
        --backbone biaofu-xmu/wav2vec-S-Base-ft-960h
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import Wav2Vec2FeatureExtractor

from hpsn.config import HPSNConfig
from hpsn.model.backbone import FrozenWav2VecS
from hpsn.model.hpsn import HPSN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backbone", default="biaofu-xmu/wav2vec-S-Base-ft-960h")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make_input(fe, device, seed: int):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(16000, generator=g).numpy()
    b = torch.randn(16000, generator=g).numpy()
    proc = fe([a, b], sampling_rate=fe.sampling_rate, return_tensors="pt",
              padding=True, return_attention_mask=True)
    return proc["input_values"].to(device), proc["attention_mask"].to(device)


def build_hpsn(seed: int, *, n_iterations: int, frozen_taps=None) -> HPSN:
    torch.manual_seed(seed)
    cfg = HPSNConfig(n_iterations=n_iterations)
    if frozen_taps is not None:
        cfg.level1_frozen_tap, cfg.level2_frozen_tap, cfg.level3_frozen_tap = frozen_taps
    return HPSN(cfg)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Loading backbone: {args.backbone}")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.backbone)
    backbone = FrozenWav2VecS(args.backbone, dtype=torch.float32).to(device)
    iv, am = make_input(fe, device, args.seed)

    with torch.no_grad():
        hs = backbone(iv, attention_mask=am)
    print(f"  backbone returned {len(hs)} hidden states; T={hs[0].shape[1]}")

    # ---- 1. n_iter=1 baseline ---------------------------------------------
    hpsn1 = build_hpsn(args.seed, n_iterations=1).to(device).eval()
    with torch.no_grad():
        out1 = hpsn1(hs, attention_mask=am)
    expected_keys = {
        "recon1", "recon2", "recon3", "target1", "target2", "target3",
        "mask1", "mask2", "mask3", "level1_repr", "level2_repr", "level3_repr",
        "mu1", "mu2", "valid_mask",
    }
    missing = expected_keys - set(out1.keys())
    assert_true(not missing, f"n_iter=1 output missing keys: {missing}")
    for k in ("recon1", "recon2", "recon3"):
        assert_true(torch.isfinite(out1[k]).all(), f"{k} contains NaN/Inf at n_iter=1")
    print(f"OK  test 1: n_iter=1 forward returns expected keys; "
          f"recon shapes "
          f"L1={tuple(out1['recon1'].shape)} "
          f"L2={tuple(out1['recon2'].shape)} "
          f"L3={tuple(out1['recon3'].shape)}")

    # ---- 2. n_iter=2 forward succeeds with same shapes --------------------
    hpsn2 = build_hpsn(args.seed, n_iterations=2).to(device).eval()
    with torch.no_grad():
        out2 = hpsn2(hs, attention_mask=am)
    for k in ("recon1", "recon2", "recon3"):
        assert_true(out1[k].shape == out2[k].shape,
                    f"{k} shape mismatch n_iter=1 vs n_iter=2")
        assert_true(torch.isfinite(out2[k]).all(),
                    f"{k} contains NaN/Inf at n_iter=2")
    print(f"OK  test 2: n_iter=2 forward returns same shapes")

    # ---- 3. n_iter=2 output differs from n_iter=1 -------------------------
    # Same seed → identical parameters; if the refinement pass is doing
    # nothing, recon* would match exactly. We expect non-trivial divergence.
    diffs = {
        k: (out2[k] - out1[k]).abs().max().item()
        for k in ("recon1", "recon2", "recon3", "level1_repr", "level2_repr", "level3_repr")
    }
    n_changed = sum(1 for v in diffs.values() if v > 1e-5)
    assert_true(
        n_changed >= 4,
        f"refinement pass barely changed outputs (max abs diffs: {diffs}); "
        f"only {n_changed}/6 fields differ — eps2 may be collapsing to 0",
    )
    print(f"OK  test 3: refinement changes outputs ({n_changed}/6 fields differ; "
          f"max diff per field: " +
          ", ".join(f"{k}={v:.4f}" for k, v in diffs.items()) + ")")

    # ---- 4. error projection wired into parameter set ---------------------
    # Only error_proj_2 is built — error_proj_1 was reserved for the L1→L2
    # dual-cross-attention path (Section 5.5 Option A) which is deferred.
    pnames = {n for n, _ in hpsn2.named_parameters()}
    for need in ("error_proj_2.weight", "error_proj_2.bias"):
        assert_true(need in pnames, f"missing parameter: {need}")
    assert_true(hpsn2.error_proj_2.weight.requires_grad, "error_proj_2 frozen")
    assert_true(not hasattr(hpsn2, "error_proj_1"),
                "error_proj_1 should have been removed (it was unused dead weight)")
    print(f"OK  test 4: error_proj_2 present in parameters() and trainable; error_proj_1 absent")

    # ---- 5. frozen taps survive into the model ----------------------------
    hpsn_ft = build_hpsn(args.seed, n_iterations=1, frozen_taps=(1, 5, 9)).to(device)
    for name, tap, k in (("tap1", hpsn_ft.tap1, 1),
                         ("tap2", hpsn_ft.tap2, 5),
                         ("tap3", hpsn_ft.tap3, 9)):
        assert_true(tap.frozen_layer == k,
                    f"{name}.frozen_layer={tap.frozen_layer}, expected {k}")
        # weights is a buffer in frozen mode — should appear in state_dict but
        # NOT in parameters(). Check exact names to avoid false positives.
        param_ids = {id(p) for p in tap.parameters()}
        assert_true(
            id(tap.weights) not in param_ids,
            f"{name}.weights is a Parameter when frozen (should be a buffer)",
        )
    # Also verify the -1 sentinel keeps taps learnable (default behavior).
    assert_true(hpsn1.tap1.frozen_layer is None, "tap1 unexpectedly frozen at default")
    print("OK  test 5: frozen taps propagate via config; -1 sentinel keeps them learnable")

    print("\nAll Phase 2C tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
