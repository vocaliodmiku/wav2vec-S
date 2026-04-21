"""Empirically measure how many future frames the Wav2VecS encoder peeks.

The block-wise attention mask in :func:`gen_block_atten_mask` lets every query
in block ``b`` attend to all keys in blocks ``0..b`` plus ``right_context``
frames taken from immediately after block ``b``. This script probes the
encoder with single-frame input perturbations and records, for every output
frame, the exact set of input frames it depends on. The maximum positive
offset ``t_in - t_out`` over that set is the empirical future-frame lookahead.

Theoretical expectation (per-query position ``q`` inside block ``b`` of size
``M = main_context`` with ``R = right_context``)::

    lookahead(q) = (b + 1) * M + R - 1 - q
                 = (M - 1 - (q mod M)) + R

so lookahead is maximal at the first frame of a block (``M - 1 + R``) and
minimal at the last frame (``R``). Stacking layers does not widen the set
because every layer applies the same mask.

Run directly::

    python tests/test_lookahead_wav2vec_s.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch

# Make the sibling ``wav2vec-S-hf`` package importable when running this file
# as a script.
_HF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wav2vec-S-hf")
if _HF_DIR not in sys.path:
    sys.path.insert(0, _HF_DIR)

from wav2vec_s.configuration_wav2vec_s import Wav2VecSConfig  # noqa: E402
from wav2vec_s.modeling_wav2vec_s import Wav2VecSEncoder  # noqa: E402


@dataclass
class LookaheadResult:
    main_context: int
    right_context: int
    seq_len: int
    per_position: list[int]   # empirical lookahead for each output frame
    theoretical: list[int]    # closed-form expectation for each output frame

    @property
    def max_lookahead(self) -> int:
        return max(self.per_position)

    @property
    def min_lookahead(self) -> int:
        return min(self.per_position)


def _build_encoder(hidden_size: int = 64, num_layers: int = 2, num_heads: int = 4) -> Wav2VecSEncoder:
    """Instantiate a tiny Wav2VecS encoder with random weights.

    Only the attention mask structure determines future-frame dependency, so
    weights do not need to be pretrained. A small config keeps the test fast
    enough to run on CPU.
    """
    config = Wav2VecSConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 2,
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        do_stable_layer_norm=False,
        context_type="constant",
        main_context=1,   # overwritten per test
        right_context=0,  # overwritten per test
    )
    # Force the eager SDPA path so attention masks are honoured exactly.
    config._attn_implementation = "eager"
    encoder = Wav2VecSEncoder(config).eval()
    # Disable any remaining stochastic paths.
    for m in encoder.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return encoder


@torch.no_grad()
def measure_lookahead(
    encoder: Wav2VecSEncoder,
    main_context: int,
    right_context: int,
    seq_len: int,
    *,
    atol: float = 1e-5,
    perturb_scale: float = 10.0,
    seed: int = 0,
) -> LookaheadResult:
    """Run single-frame perturbations and derive the dependency set per output.

    For every input frame ``t_in`` we run a forward pass with that frame (and
    only that frame) replaced by a large random vector. Any output frame
    ``t_out`` whose activation changes depends on ``t_in``. The per-position
    lookahead is ``max(t_in - t_out)`` over the affected set, clipped at 0.
    """
    encoder.main_context = main_context
    encoder.right_context = right_context

    device = next(encoder.parameters()).device
    dtype = next(encoder.parameters()).dtype
    d = encoder.config.hidden_size

    # Make ``seq_len`` a clean multiple of ``main_context`` so there are no
    # ragged trailing frames (``gen_block_atten_mask`` uses ``seq_len // M``).
    seq_len = (seq_len // main_context) * main_context
    assert seq_len > 0, f"seq_len must be >= main_context (got M={main_context})"

    g = torch.Generator(device="cpu").manual_seed(seed)
    x_base = torch.randn(1, seq_len, d, generator=g).to(device=device, dtype=dtype)

    y_base = encoder(x_base.clone(), return_dict=True).last_hidden_state

    affected_by_input: list[set[int]] = [set() for _ in range(seq_len)]  # indexed by t_out
    for t_in in range(seq_len):
        perturb = torch.randn(d, generator=g).to(device=device, dtype=dtype) * perturb_scale
        x_pert = x_base.clone()
        x_pert[0, t_in] = x_pert[0, t_in] + perturb
        y_pert = encoder(x_pert, return_dict=True).last_hidden_state

        delta = (y_pert - y_base).abs().amax(dim=-1).squeeze(0)  # (seq_len,)
        changed = (delta > atol).nonzero(as_tuple=False).flatten().tolist()
        for t_out in changed:
            affected_by_input[t_out].add(t_in)

    per_position: list[int] = []
    for t_out in range(seq_len):
        deps = affected_by_input[t_out]
        future = [t_in - t_out for t_in in deps if t_in > t_out]
        per_position.append(max(future) if future else 0)

    theoretical = [
        (main_context - 1 - (q % main_context)) + right_context for q in range(seq_len)
    ]
    return LookaheadResult(
        main_context=main_context,
        right_context=right_context,
        seq_len=seq_len,
        per_position=per_position,
        theoretical=theoretical,
    )


def _format_result(r: LookaheadResult) -> str:
    head = (
        f"M={r.main_context:>3d}  R={r.right_context:>3d}  T={r.seq_len:>3d}  "
        f"max_lookahead={r.max_lookahead:>3d}  min_lookahead={r.min_lookahead:>3d}"
    )
    # Show the first block's per-position profile so the shape is visible.
    block = r.per_position[: r.main_context]
    expect = r.theoretical[: r.main_context]
    return f"{head}\n    first-block empirical:  {block}\n    first-block theoretical:{expect}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_sweep(configs, seq_len_target: int = 48) -> list[LookaheadResult]:
    encoder = _build_encoder()
    results: list[LookaheadResult] = []
    for main_context, right_context in configs:
        r = measure_lookahead(
            encoder,
            main_context=main_context,
            right_context=right_context,
            seq_len=seq_len_target,
        )
        results.append(r)
    return results


def test_lookahead_matches_closed_form():
    """Empirical lookahead must equal ``M - 1 - (q mod M) + R`` everywhere."""
    configs = [
        (1, 0),   # fully causal, zero lookahead
        (4, 0),   # block-causal, no right context
        (8, 2),   # the configuration from example.py
        (8, 4),
        (16, 8),  # default in Wav2VecSConfig
    ]
    for r in _run_sweep(configs):
        assert r.per_position == r.theoretical, (
            f"mismatch for M={r.main_context}, R={r.right_context}:\n"
            f"  empirical:   {r.per_position}\n"
            f"  theoretical: {r.theoretical}"
        )


def test_max_lookahead_formula():
    """The largest lookahead across positions is ``M - 1 + R``."""
    for r in _run_sweep([(1, 0), (4, 0), (8, 2), (8, 4), (16, 8)]):
        assert r.max_lookahead == r.main_context - 1 + r.right_context, _format_result(r)
        assert r.min_lookahead == r.right_context, _format_result(r)


def test_fully_causal_when_right_context_zero_and_main_context_one():
    """``M=1, R=0`` degenerates to a strictly frame-causal encoder."""
    (r,) = _run_sweep([(1, 0)])
    assert r.max_lookahead == 0, _format_result(r)


if __name__ == "__main__":
    configs = [
        (1, 0),
        (4, 0),
        (4, 2),
        (8, 0),
        (8, 2),   # example.py setting
        (8, 4),
        (16, 0),
        (16, 8),  # Wav2VecSConfig default
    ]
    print("Empirical future-frame lookahead for the Wav2VecS block-wise encoder")
    print("(one forward pass per perturbed input frame; weights are random but")
    print(" only the attention mask determines dependency structure)\n")
    for r in _run_sweep(configs):
        print(_format_result(r))
        print()
    print("OK: lookahead measurements complete")
