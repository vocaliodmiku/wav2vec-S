"""Test whether Wav2VecSModel is causal.

Perturb CNN features after a split frame and check how much it affects encoder
outputs at earlier positions. A causal (or block-causal) model should leave
prefix outputs unchanged; a bidirectional model should not.
"""

import torch
from wav2vec_s.modeling_wav2vec_s import Wav2VecSModel


torch.manual_seed(0)

model_path = "biaofu-xmu/wav2vec-S-Base-ft-960h"
model = Wav2VecSModel.from_pretrained(model_path).to(torch.float32).cuda().eval()

main_context = model.config.main_context
trained_right_context = 1

# random waveform long enough for several full blocks on either side of the split
waveform = torch.randn(1, 16000 * 8, device="cuda")
with torch.no_grad():
    feats = model.extract_cnn_features(waveform)

T = feats.size(1)
split = (T // 2 // main_context) * main_context  # split on a block boundary

feats_a = feats.clone()
feats_b = feats.clone()
feats_b[:, split:] = torch.randn_like(feats_b[:, split:])

print(f"total frames: {T}, split frame: {split}, main_context: {main_context}")

for right_context in [trained_right_context, 0]:
    model.encoder.right_context = right_context
    with torch.no_grad():
        out_a = model.forward_encoder(feats_a).last_hidden_state
        out_b = model.forward_encoder(feats_b).last_hidden_state

    # A query frame at position q (block b_q = q // M) can attend to key positions
    # up to (b_q + 1) * M + right_context - 1. To be unaffected by features at
    # index >= split, we need (b_q + 1) * M + right_context <= split.
    # Add one block of safety margin to absorb any numerical leakage (layer norm,
    # positional embedding conv, etc.).
    offset = right_context + main_context
    safe_end = max(split - offset, 0)
    safe_end = (safe_end // main_context) * main_context

    if safe_end == 0:
        raise RuntimeError(
            f"No safe prefix frames to compare (split={split}, offset={offset}). "
            "Use a longer waveform."
        )

    prefix_diff = (out_a[:, :safe_end] - out_b[:, :safe_end]).abs().max().item()
    tail_diff = (out_a[:, split:] - out_b[:, split:]).abs().max().item()

    verdict = "CAUSAL (prefix unaffected)" if prefix_diff < 1e-4 else "NOT CAUSAL"
    print(
        f"right_context={right_context:2d} offset={offset:2d} | "
        f"prefix[0:{safe_end}] max diff = {prefix_diff:.3e} "
        f"| tail[{split}:] max diff = {tail_diff:.3e} -> {verdict}"
    )


# Block-causal vs token-causal: perturb in the middle of a block and check
# whether earlier frames within the same block are affected.
print("\n--- token-causal check (within-block perturbation) ---")
mid_block_split = split + main_context // 2
feats_c = feats.clone()
feats_d = feats.clone()
feats_d[:, mid_block_split:] = torch.randn_like(feats_d[:, mid_block_split:])

model.encoder.right_context = 0
with torch.no_grad():
    out_c = model.forward_encoder(feats_c).last_hidden_state
    out_d = model.forward_encoder(feats_d).last_hidden_state

same_block_prefix_diff = (
    out_c[:, split:mid_block_split] - out_d[:, split:mid_block_split]
).abs().max().item()
print(
    f"perturbed features at frame {mid_block_split} (inside block starting at {split}); "
    f"frames [{split}:{mid_block_split}] max diff = {same_block_prefix_diff:.3e}"
)
print(
    "-> block-causal (not token-causal)" if same_block_prefix_diff > 1e-4
    else "-> token-causal"
)
