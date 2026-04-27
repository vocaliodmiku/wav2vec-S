"""ARPAbet → phonological-feature lookup for HPSN-v2 L1 targets.

The 14-feature articulatory representation below is the standard set used in
phonology textbooks (Hayes 2009; Ladefoged & Johnson 2014). Each phone maps to
a binary vector; the lookup is deterministic, offline, and requires no
external model.

Stress digits on vowels (AA0/AA1/AA2) are stripped before lookup.

Special labels:
  ""        — empty interval, silence
  "sil"     — silence
  "sp"      — short pause
  "spn"     — spurious noise
  "<unk>"   — unknown
All map to the all-zero vector and ID 0 ("SIL").
"""
from __future__ import annotations

import numpy as np

# 14 binary articulatory features
FEATURE_NAMES: tuple[str, ...] = (
    # manner (7)
    "vowel", "stop", "fricative", "affricate", "nasal", "liquid", "glide",
    # place (6)
    "bilabial", "labiodental", "alveolar", "palatal", "velar", "glottal",
    # voicing (1)
    "voiced",
)

N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 14


def _f(**kw) -> np.ndarray:
    """Build a feature vector from named flags (default 0)."""
    v = np.zeros(N_FEATURES, dtype=np.float32)
    for name, val in kw.items():
        v[FEATURE_NAMES.index(name)] = float(val)
    return v


# fmt: off
PHONE_FEATURES: dict[str, np.ndarray] = {
    # ── vowels ── all voiced; place is moot, leave place flags 0
    "AA": _f(vowel=1, voiced=1),
    "AE": _f(vowel=1, voiced=1),
    "AH": _f(vowel=1, voiced=1),
    "AO": _f(vowel=1, voiced=1),
    "AW": _f(vowel=1, voiced=1),
    "AY": _f(vowel=1, voiced=1),
    "EH": _f(vowel=1, voiced=1),
    "ER": _f(vowel=1, voiced=1),
    "EY": _f(vowel=1, voiced=1),
    "IH": _f(vowel=1, voiced=1),
    "IY": _f(vowel=1, voiced=1),
    "OW": _f(vowel=1, voiced=1),
    "OY": _f(vowel=1, voiced=1),
    "UH": _f(vowel=1, voiced=1),
    "UW": _f(vowel=1, voiced=1),

    # ── plosives ──
    "P":  _f(stop=1, bilabial=1),
    "B":  _f(stop=1, bilabial=1, voiced=1),
    "T":  _f(stop=1, alveolar=1),
    "D":  _f(stop=1, alveolar=1, voiced=1),
    "K":  _f(stop=1, velar=1),
    "G":  _f(stop=1, velar=1, voiced=1),

    # ── fricatives ──
    "F":  _f(fricative=1, labiodental=1),
    "V":  _f(fricative=1, labiodental=1, voiced=1),
    "TH": _f(fricative=1, alveolar=1),       # dental ≈ alveolar in this 14-feat scheme
    "DH": _f(fricative=1, alveolar=1, voiced=1),
    "S":  _f(fricative=1, alveolar=1),
    "Z":  _f(fricative=1, alveolar=1, voiced=1),
    "SH": _f(fricative=1, palatal=1),         # postalveolar ≈ palatal here
    "ZH": _f(fricative=1, palatal=1, voiced=1),
    "HH": _f(fricative=1, glottal=1),

    # ── affricates ──
    "CH": _f(affricate=1, palatal=1),
    "JH": _f(affricate=1, palatal=1, voiced=1),

    # ── nasals (all voiced) ──
    "M":  _f(nasal=1, bilabial=1, voiced=1),
    "N":  _f(nasal=1, alveolar=1, voiced=1),
    "NG": _f(nasal=1, velar=1, voiced=1),

    # ── liquids ──
    "L":  _f(liquid=1, alveolar=1, voiced=1),
    "R":  _f(liquid=1, alveolar=1, voiced=1),

    # ── glides ──
    "W":  _f(glide=1, bilabial=1, voiced=1),
    "Y":  _f(glide=1, palatal=1, voiced=1),
}
# fmt: on


# Phone-ID table: 0 reserved for silence/unknown.
# Order is the canonical ARPAbet ordering used by CMUDict.
PHONE_VOCAB: tuple[str, ...] = (
    "SIL",  # 0 — silence/unknown sentinel (must be index 0)
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH",
    "EH", "ER", "EY",
    "F", "G", "HH",
    "IH", "IY",
    "JH", "K", "L", "M", "N", "NG",
    "OW", "OY",
    "P", "R", "S", "SH", "T", "TH",
    "UH", "UW", "V", "W", "Y", "Z", "ZH",
)
PHONE_TO_ID: dict[str, int] = {p: i for i, p in enumerate(PHONE_VOCAB)}
assert PHONE_TO_ID["SIL"] == 0
N_PHONES = len(PHONE_VOCAB)  # 40

_SILENCE_ALIASES: frozenset[str] = frozenset(
    {"", "sil", "sp", "spn", "<unk>", "unk", "noise"}
)


def normalize_phone(label: str) -> str:
    """Strip stress digits, casefold, map silence-likes to ``"SIL"``.

    >>> normalize_phone("AA1")
    'AA'
    >>> normalize_phone("")
    'SIL'
    """
    s = (label or "").strip()
    if s.lower() in _SILENCE_ALIASES:
        return "SIL"
    s = s.upper()
    while s and s[-1].isdigit():
        s = s[:-1]
    return s or "SIL"


def phone_to_id(label: str) -> int:
    """Return the integer ID for an ARPAbet label (0 for silence/unknown)."""
    return PHONE_TO_ID.get(normalize_phone(label), 0)


def phone_to_features(label: str) -> np.ndarray:
    """Return the 14-d articulatory feature vector (zeros for silence/unknown)."""
    canon = normalize_phone(label)
    if canon == "SIL" or canon not in PHONE_FEATURES:
        return np.zeros(N_FEATURES, dtype=np.float32)
    return PHONE_FEATURES[canon].copy()
