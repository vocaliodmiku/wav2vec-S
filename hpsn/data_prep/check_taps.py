"""HPSN-v2 Phase 2B smoke test for the frozen-one-hot ``LayerTap``.

Asserts the invariants HPSN-v2 will rely on when taps are pinned to a
specific backbone layer per band:

1. ``frozen_layer`` validation rejects out-of-band values.
2. Learnable mode still works (default).
3. Frozen-one-hot mode produces output exactly equal to the requested
   backbone hidden state (within fp32/bf16 tolerance).
4. ``weights`` is a buffer (not a Parameter) in frozen mode → not in
   ``parameters()`` and untouched by the optimizer.
5. State dict round-trips: save → load preserves the one-hot weights.

Usage
-----
    python -m hpsn.data_prep.check_taps \\
        --backbone biaofu-xmu/wav2vec-S-Base-ft-960h
"""
from __future__ import annotations

import argparse
import io
import sys

import torch
from transformers import Wav2Vec2FeatureExtractor

from hpsn.model.backbone import FrozenWav2VecS, LayerTap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backbone", default="biaofu-xmu/wav2vec-S-Base-ft-960h")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    # ---- 1. validation ------------------------------------------------------
    try:
        LayerTap(layers=(1, 2, 3, 4), frozen_layer=99)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for frozen_layer not in layers")
    print("OK  test 1: rejects out-of-band frozen_layer")

    # ---- 4. parameter vs buffer --------------------------------------------
    learn_tap = LayerTap(layers=(1, 2, 3, 4))
    frozen_tap = LayerTap(layers=(1, 2, 3, 4), frozen_layer=1)
    assert_true(
        any(p is learn_tap.weights for p in learn_tap.parameters()),
        "learnable tap should expose weights as a Parameter",
    )
    assert_true(
        not any(p is frozen_tap.weights for p in frozen_tap.parameters()),
        "frozen tap should NOT expose weights as a Parameter",
    )
    assert_true(
        torch.equal(frozen_tap.weights, torch.tensor([1.0, 0.0, 0.0, 0.0])),
        f"unexpected frozen weights: {frozen_tap.weights}",
    )
    print("OK  test 4: weights is a Parameter when learnable, a buffer when frozen")

    # ---- 5. state-dict round trip ------------------------------------------
    buf = io.BytesIO()
    torch.save(frozen_tap.state_dict(), buf)
    buf.seek(0)
    new_tap = LayerTap(layers=(1, 2, 3, 4), frozen_layer=1)
    new_tap.load_state_dict(torch.load(buf, weights_only=True))
    assert_true(
        torch.equal(new_tap.weights, frozen_tap.weights),
        "state dict round-trip changed frozen weights",
    )
    print("OK  test 5: state dict round-trip preserves frozen weights")

    # ---- 2 & 3. real backbone ----------------------------------------------
    print(f"\nLoading backbone: {args.backbone}")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.backbone)
    backbone = FrozenWav2VecS(args.backbone, dtype=torch.float32).to(device)

    # Synthetic 1-second batch
    B, T_sec = 2, 1.0
    n = int(fe.sampling_rate * T_sec)
    proc = fe(
        [torch.randn(n).numpy(), torch.randn(n).numpy()],
        sampling_rate=fe.sampling_rate,
        return_tensors="pt", padding=True, return_attention_mask=True,
    )
    iv = proc["input_values"].to(device)
    am = proc["attention_mask"].to(device)
    with torch.no_grad():
        hs = backbone(iv, attention_mask=am)
    print(f"  backbone produced {len(hs)} hidden states (layers 0..{len(hs)-1})")

    # Test 2: learnable mode runs; output is finite.
    learn_tap.to(device)
    out_learn = learn_tap(list(hs))
    assert_true(torch.isfinite(out_learn).all(), "learnable tap produced NaN/Inf")
    print(f"OK  test 2: learnable tap output shape={tuple(out_learn.shape)}, "
          f"max|val|={out_learn.abs().max().item():.4f}")

    # Test 3: frozen tap on layer k equals hidden_states[k] in fp32.
    for k in (1, 2, 3, 4):
        ftap = LayerTap(layers=(1, 2, 3, 4), frozen_layer=k).to(device)
        out_frozen = ftap(list(hs))
        ref = hs[k].float()
        diff = (out_frozen - ref).abs().max().item()
        assert_true(
            diff < 1e-5,
            f"frozen_layer={k}: output not equal to hidden_states[{k}] "
            f"(max abs diff = {diff:.2e})",
        )
    print(f"OK  test 3: frozen tap for k ∈ {{1,2,3,4}} equals hidden_states[k] (max diff < 1e-5)")

    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
