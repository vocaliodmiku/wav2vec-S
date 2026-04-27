Here's a concise end-to-end procedure for the MEG-MASC evaluation.

## Step 0 — Get the data

MEG-MASC provides MEG recordings of 27 English speakers who listened to two hours of naturalistic stories across two identical sessions. The data is organized in BIDS format with word and phoneme onset/offset timestamps already aligned to the audio. The code and data pipeline are available at `github.com/kingjr/masc-bids`, and you'll use MNE-Python for all MEG processing.

## Step 1 — Preprocess MEG

Using MNE-Python on the raw BIDS data:

**Filtering**: bandpass 0.5–40 Hz (captures ERF components, removes line noise and slow drift). Apply notch filter at 60 Hz if US data.

**Artifact rejection**: use ICA to remove eye blinks and heartbeat artifacts. MEG-MASC includes EOG and ECG channels for this. Typically 1-3 ICA components removed per subject.

**Downsampling**: downsample to 100 Hz or 120 Hz. This is important — your wav2vec2 outputs at 50 Hz (one frame per 20ms), so you need to choose a common sampling rate. Two options: downsample MEG to 50 Hz to match wav2vec2 directly, or keep MEG at 100 Hz and upsample wav2vec2 features (linear interpolation). I'd recommend downsampling MEG to 50 Hz for simplicity — it preserves all relevant neural frequencies up to 25 Hz, which covers the ERF components you care about (N100, P200, N400).

**Epoching for encoding model**: you do NOT epoch into trials. Instead, keep the continuous MEG signal aligned to the continuous audio. The encoding model operates on the full time series, not on averaged evoked responses. This is key — you're doing single-trial encoding, not ERP analysis.

**Sensor selection**: use all magnetometer or gradiometer sensors (typically 204 gradiometers or 102 magnetometers on Neuromag systems). You can also run sensor-space analysis on subsets: temporal sensors (auditory cortex), frontal sensors (prediction/semantic), to test spatial predictions of the HPSN model.

```python
import mne
from mne_bids import read_raw_bids

# Load one subject
raw = read_raw_bids(bids_path)
raw.load_data()

# Preprocess
raw.filter(0.5, 40.0)
raw.notch_filter(60.0)

# ICA artifact removal
ica = mne.preprocessing.ICA(n_components=20)
ica.fit(raw)
# Find and exclude EOG/ECG components
eog_indices, _ = ica.find_bads_eog(raw)
ecg_indices, _ = ica.find_bads_ecg(raw)
ica.exclude = eog_indices + ecg_indices
ica.apply(raw)

# Downsample to match wav2vec2 frame rate
raw.resample(50)  # 50 Hz = 20ms per sample = wav2vec2 frame rate
meg_data = raw.get_data(picks='grad')  # [n_sensors, n_timepoints]
```

## Step 2 — Extract model representations aligned to the audio

Run the same audio stimuli through your frozen wav2vec2 and then through HPSN. The critical requirement: the model's time axis must be perfectly aligned to the MEG's time axis.

```python
def extract_aligned_representations(audio_path, backbone, hpsn):
    """
    Process stimulus audio through models.
    wav2vec2 outputs at 50 Hz (one frame per 20ms).
    MEG is downsampled to 50 Hz.
    So frame indices align directly.
    """
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)

    with torch.no_grad():
        hidden_states = backbone(audio)

        # Baseline representations (raw wav2vec2)
        baseline_low = hpsn.tap_acoustic(hidden_states)   # [1, T, 768]
        baseline_mid = hpsn.tap_lexical(hidden_states)     # [1, T, 768]

        # HPSN representations (with top-down processing)
        outputs = hpsn(hidden_states)  # masking disabled in eval mode
        hpsn_l1 = outputs['level1_repr']  # [1, T, 512]
        hpsn_l2 = outputs['level2_repr']  # [1, T, 512]

    return {
        'baseline_low': baseline_low.squeeze(0).numpy(),   # [T, 768]
        'baseline_mid': baseline_mid.squeeze(0).numpy(),   # [T, 768]
        'hpsn_l1': hpsn_l1.squeeze(0).numpy(),             # [T, 512]
        'hpsn_l2': hpsn_l2.squeeze(0).numpy(),             # [T, 512]
    }
```

**Alignment**: MEG-MASC provides precise onset times for each audio stimulus within the MEG recording. Use these to crop the MEG data to exactly match the model output time axis. Both are at 50 Hz, so frame indices correspond one-to-one.

## Step 3 — Fit encoding models (ridge regression)

For each subject, each sensor, each model condition: fit a ridge regression from model representations to MEG signal, using temporal lags.

**Why temporal lags?** The brain response to a speech frame at time t doesn't happen only at time t — it unfolds over ~500ms. A frame of speech at t=0 produces an N100 at t=100ms, processing at t=200ms, semantic integration at t=400ms. The lagged regression captures this temporal spread.

```python
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

def build_lagged_matrix(features, lags):
    """
    Create time-lagged feature matrix.

    Args:
        features: [T, D] — model representations
        lags: list of int — lag values in frames (at 50 Hz, 1 frame = 20ms)
            e.g., [0, 1, 2, 3, 5, 7, 10, 15, 20, 25]
            covers 0ms to 500ms post-stimulus

    Returns:
        X: [T, D * n_lags] — lagged feature matrix
    """
    T, D = features.shape
    n_lags = len(lags)
    X = np.zeros((T, D * n_lags))

    for i, lag in enumerate(lags):
        if lag >= 0:
            X[lag:, i*D:(i+1)*D] = features[:T-lag]
        else:
            X[:T+lag, i*D:(i+1)*D] = features[-lag:]

    return X


def encoding_model(features, meg, lags, n_splits=5):
    """
    Fit ridge regression encoding model with cross-validation.

    Args:
        features: [T, D] model representations
        meg: [T, n_sensors] MEG data
        lags: list of lag values in frames
        n_splits: number of CV folds

    Returns:
        r_values: [n_sensors] Pearson r per sensor (averaged across folds)
        r_by_lag: [n_lags, n_sensors] correlation contribution per lag
    """
    X = build_lagged_matrix(features, lags)

    # Trim edges where lag padding creates zeros
    max_lag = max(max(lags), abs(min(lags)))
    X = X[max_lag:-max_lag]
    meg = meg[max_lag:-max_lag]

    # Standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    meg = (meg - meg.mean(0)) / (meg.std(0) + 1e-8)

    # Cross-validated ridge regression
    # IMPORTANT: use contiguous blocks, not random splits
    # (temporal autocorrelation would inflate scores with random splits)
    kf = KFold(n_splits=n_splits, shuffle=False)

    r_values = np.zeros(meg.shape[1])
    n_folds = 0

    for train_idx, test_idx in kf.split(X):
        ridge = RidgeCV(alphas=np.logspace(-1, 6, 20))
        ridge.fit(X[train_idx], meg[train_idx])
        pred = ridge.predict(X[test_idx])

        # Pearson r per sensor
        for s in range(meg.shape[1]):
            r = np.corrcoef(pred[:, s], meg[test_idx, s])[0, 1]
            r_values[s] += r
        n_folds += 1

    r_values /= n_folds
    return r_values
```

**Key methodological points**:

Use **contiguous block cross-validation**, never random splits. MEG data is temporally autocorrelated — random splits leak information and inflate scores.

Use **RidgeCV** with a broad alpha range (0.1 to 1e6). The optimal regularization varies across sensors and subjects.

Typical lag range: 0 to 500ms post-stimulus (0 to 25 frames at 50 Hz). You can include negative lags (-100ms to 0) as a control — these should show near-zero correlation since the brain can't respond before the stimulus.

## Step 4 — Run the comparison

```python
# For each subject
lags = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25]  # 0ms to 500ms

conditions = {
    'wav2vec2_low':  reprs['baseline_low'],    # Layers 1-4
    'wav2vec2_mid':  reprs['baseline_mid'],    # Layers 5-8
    'hpsn_level1':   reprs['hpsn_l1'],         # HPSN Level 1 (with top-down)
    'hpsn_level2':   reprs['hpsn_l2'],         # HPSN Level 2 (with inhibition)
    'hpsn_l1_no_td': reprs['hpsn_l1_ablated'], # Level 1 (top-down cut)
}

results = {}
for name, features in conditions.items():
    r = encoding_model(features, meg_data.T, lags)
    results[name] = r  # [n_sensors]
```

## Step 5 — Statistical testing

```python
from scipy.stats import wilcoxon

# Compare HPSN Level 1 vs wav2vec2 baseline (across 27 subjects)
# Each subject gives one average r value
hpsn_scores = [results_per_subject[s]['hpsn_level1'].mean() for s in subjects]
baseline_scores = [results_per_subject[s]['wav2vec2_low'].mean() for s in subjects]

# Wilcoxon signed-rank test (paired, non-parametric)
stat, p_value = wilcoxon(hpsn_scores, baseline_scores, alternative='greater')

# Effect size: median improvement
improvement = np.array(hpsn_scores) - np.array(baseline_scores)
print(f"Median Δr = {np.median(improvement):.4f}, p = {p_value:.4f}")
```

Note that a Pearson R around 0.08 is typical for single-trial MEG data during continuous listening, so even small improvements (Δr = 0.005-0.01) are meaningful if consistent across subjects.

## Step 6 — The time-resolved analysis (the key figure)

This is where you test whether the top-down contribution appears at *early* time windows:

```python
def encoding_per_lag(features, meg, lags):
    """
    Fit a SEPARATE encoding model for each lag.
    This tells you WHEN in the brain response each model contributes.
    """
    results = {}
    for lag in lags:
        X = build_lagged_matrix(features, [lag])  # single lag
        # ... same ridge CV as above ...
        results[lag] = r_values  # [n_sensors]
    return results

# For each subject, compute per-lag encoding for both conditions
for s in subjects:
    per_lag_hpsn = encoding_per_lag(hpsn_l1, meg, lags)
    per_lag_baseline = encoding_per_lag(baseline_low, meg, lags)

    # The improvement curve
    improvement_by_lag = {
        lag: per_lag_hpsn[lag].mean() - per_lag_baseline[lag].mean()
        for lag in lags
    }
    # Plot this: x-axis = lag (ms), y-axis = Δr
    # Prediction: peak improvement at lags 2-7 (40-140ms)
    # = evidence for top-down modulation of early auditory processing
```

**What to plot**: the improvement curve (HPSN minus baseline) as a function of temporal lag. If your top-down mechanism is working, this curve should show a peak in the 50-200ms window — meaning HPSN representations capture something about early auditory processing that raw wav2vec2 misses. That "something" is the top-down modulation.

## Step 7 — Sensor-space analysis (bonus spatial prediction)

Group sensors by brain region to test spatial predictions:

```python
# Neuromag sensor layout — rough groupings:
temporal_sensors = [...]   # Over auditory cortex
frontal_sensors = [...]    # Over prefrontal cortex
parietal_sensors = [...]   # Over parietal cortex

# Prediction: HPSN Level 1 improvement should be strongest
# over temporal sensors (early auditory processing)
# HPSN Level 2 improvement should spread to frontal sensors
# (lexical competition, semantic prediction)
```

## Summary of what to report

Your results section needs four things: (1) overall R² or Pearson r for each condition averaged across subjects and sensors, with significance tests, (2) the improvement curve across temporal lags showing *when* the top-down contribution matters, (3) the ablation comparison (full HPSN vs. top-down-cut) proving the gain comes specifically from the top-down pathway, and (4) the sensor-space map showing *where* the improvement is localized. If all four converge — top-down helps, the gain is early, cutting top-down removes the gain, and the effect is strongest over auditory cortex — that's a very tight story.