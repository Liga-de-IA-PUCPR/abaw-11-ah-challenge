"""Testes de `src.features.asr_timing` — 16 features de tempo apagado pelo ASR."""

from __future__ import annotations

import math

import pytest

from src.features.asr_timing import (
    ASR_TIMING_FEATURE_NAMES,
    N_ASR_TIMING_FEATURES,
    extract_asr_timing_features,
    extract_asr_timing_vector,
)


def _chunk(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text, "language": "en"}


def _assert_valid(feats: dict[str, float]) -> None:
    assert list(feats.keys()) == ASR_TIMING_FEATURE_NAMES
    assert len(feats) == 16 == N_ASR_TIMING_FEATURES
    for name, value in feats.items():
        assert math.isfinite(value), f"{name} não é finito: {value}"


def test_zero_chunks_returns_zero_vector():
    feats = extract_asr_timing_features([])
    _assert_valid(feats)
    assert all(v == 0.0 for v in feats.values())


def test_one_chunk_returns_zero_gap_features_but_no_crash():
    feats = extract_asr_timing_features([_chunk(0.0, 1.5, "hello there")])
    _assert_valid(feats)
    assert feats["asr_gap_count"] == 0.0
    assert feats["asr_chunk_dur_mean"] == pytest.approx(1.5)


def test_normal_gaps_counted_correctly():
    chunks = [
        _chunk(0.0, 1.0, "hello"),
        _chunk(1.5, 2.5, "world"),  # gap 0.5
        _chunk(3.5, 4.0, "again"),  # gap 1.0
    ]
    feats = extract_asr_timing_features(chunks)
    _assert_valid(feats)
    assert feats["asr_gap_count"] == 2.0
    assert feats["asr_gap_total_time"] == pytest.approx(1.5)
    assert feats["asr_gap_mean"] == pytest.approx(0.75)
    assert feats["asr_gap_max"] == pytest.approx(1.0)
    assert feats["asr_gap_count_per_chunk"] == pytest.approx(2 / 3)


def test_whisper_30s_reset_not_counted_as_gap_or_negative():
    # Chunk 2 "reseta" o relógio interno: start (2.0) <= end do anterior (28.0).
    chunks = [
        _chunk(0.0, 28.0, "long chunk before reset"),
        _chunk(2.0, 5.0, "after reset"),
        _chunk(6.0, 7.0, "normal gap after"),  # gap real 1.0 vs chunk anterior
    ]
    feats = extract_asr_timing_features(chunks)
    _assert_valid(feats)
    # Só o par (2, 3) é monotônico -> 1 gap real, valor pequeno e positivo.
    assert feats["asr_gap_count"] == 1.0
    assert feats["asr_gap_total_time"] == pytest.approx(1.0)
    assert feats["asr_gap_total_time"] >= 0.0
    assert feats["asr_gap_max"] < 20.0  # não deve "vazar" o reset como gap gigante


def test_no_gaps_at_all_contiguous_chunks():
    chunks = [
        _chunk(0.0, 1.0, "a"),
        _chunk(1.0, 2.0, "b"),
        _chunk(2.0, 3.0, "c"),
    ]
    feats = extract_asr_timing_features(chunks)
    _assert_valid(feats)
    assert feats["asr_gap_count"] == 0.0
    assert feats["asr_gap_total_time"] == 0.0
    assert feats["asr_gap_mean"] == 0.0
    assert feats["asr_gap_max"] == 0.0
    assert feats["asr_gap_before_contrastive_ratio"] == 0.0


def test_gap_before_contrastive_word_detected():
    chunks = [
        _chunk(0.0, 1.0, "I think it is good"),
        _chunk(2.0, 3.0, "but I am not sure"),  # gap 1.0 before "but"
    ]
    feats = extract_asr_timing_features(chunks)
    _assert_valid(feats)
    assert feats["asr_gap_count"] == 1.0
    assert feats["asr_gap_before_contrastive_ratio"] == pytest.approx(1.0)
    assert feats["asr_gap_before_contrastive_mean"] == pytest.approx(1.0)


def test_no_contrastive_words_found():
    chunks = [
        _chunk(0.0, 1.0, "I think it is good"),
        _chunk(2.0, 3.0, "and it works well"),
    ]
    feats = extract_asr_timing_features(chunks)
    _assert_valid(feats)
    assert feats["asr_gap_before_contrastive_ratio"] == 0.0
    assert feats["asr_gap_before_contrastive_mean"] == 0.0


def test_restart_chunk_after_gap_detected():
    chunks = [
        _chunk(0.0, 1.0, "well I guess"),
        _chunk(1.8, 2.9, "uh"),  # gap 0.8, short chunk (1.1s > 0.3 default!)
    ]
    # Make the second chunk genuinely short (< 0.3s) to trigger a restart.
    chunks = [
        _chunk(0.0, 1.0, "well I guess"),
        _chunk(1.8, 2.0, "uh"),  # gap 0.8, duration 0.2s < 0.3 threshold
    ]
    feats = extract_asr_timing_features(chunks)
    _assert_valid(feats)
    assert feats["asr_restart_count"] == 1.0
    assert feats["asr_restart_count_per_chunk"] == pytest.approx(0.5)


def test_restart_threshold_is_configurable():
    chunks = [
        _chunk(0.0, 1.0, "well I guess"),
        _chunk(1.8, 2.4, "uh"),  # duration 0.6s
    ]
    feats_default = extract_asr_timing_features(chunks)
    assert feats_default["asr_restart_count"] == 0.0

    feats_wide = extract_asr_timing_features(chunks, restart_max_dur_s=1.0)
    assert feats_wide["asr_restart_count"] == 1.0


def test_vector_matches_dict_order():
    chunks = [
        _chunk(0.0, 1.0, "hello"),
        _chunk(1.5, 2.5, "world"),
    ]
    feats = extract_asr_timing_features(chunks)
    vec = extract_asr_timing_vector(chunks)
    assert vec.shape == (16,)
    for i, name in enumerate(ASR_TIMING_FEATURE_NAMES):
        assert vec[i] == pytest.approx(feats[name])


def test_feature_names_are_stable_and_unique():
    assert len(ASR_TIMING_FEATURE_NAMES) == 16
    assert len(set(ASR_TIMING_FEATURE_NAMES)) == 16
    assert all(isinstance(n, str) for n in ASR_TIMING_FEATURE_NAMES)
