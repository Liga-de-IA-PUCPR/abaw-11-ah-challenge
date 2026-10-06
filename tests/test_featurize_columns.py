"""Testes das colunas extras do plano MoE (featurize_columns + marcadores de hesitação)."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from src.data.schema import VideoRecord, WindowSample
from src.data.windowing import save_window_index
from src.features import columns as columns_mod
from src.features.hesitation_markers import (
    HESITATION_MARKER_NAMES,
    N_HESITATION_MARKERS,
    extract_hesitation_markers,
)
from src.pipeline.featurize_columns import merge_window_columns, run_featurize_columns

# ---- marcadores --------------------------------------------------------------


def test_markers_are_rates_per_100_words():
    feats = extract_hesitation_markers("I think I I was not sure, but um I was I was fine")
    assert list(feats) == HESITATION_MARKER_NAMES and N_HESITATION_MARKERS == 11
    n_words = 14
    assert feats["hm_hedge_epistemic"] == pytest.approx(100 / n_words)  # "i think"
    assert feats["hm_uncertainty"] == pytest.approx(100 / n_words)  # "not sure"
    assert feats["hm_negation"] == pytest.approx(100 / n_words)  # "not"
    assert feats["hm_contrast"] == pytest.approx(100 / n_words)  # "but"
    assert feats["hm_filled_pause"] == pytest.approx(100 / n_words)  # "um"
    assert feats["hm_word_repetition"] == pytest.approx(100 / n_words)  # "I I"
    assert feats["hm_phrase_repetition"] == pytest.approx(100 / n_words)  # "I was I was"


def test_markers_empty_text_is_zero():
    assert set(extract_hesitation_markers("").values()) == {0.0}


# ---- pipeline de colunas -----------------------------------------------------


def _windows() -> list[WindowSample]:
    return [
        WindowSample(f"{v}#w{i}", v, v, 2.5 * i, 2.5 * i + 5.0, "txt", 0, "neutral", "train", 1)
        for v, n in [("v1", 3), ("v2", 2)]
        for i in range(n)
    ]


class _VideoLen(columns_mod.VideoLevelColumn):
    """Coluna de vídeo falsa: [comprimento da transcrição] (checa o broadcast)."""

    column = "fake_len"

    def compute_video(self, record):
        return [float(len(record.full_transcript))]

    def feature_names(self):
        return ["fake_len_0"]


class _WindowIdx(columns_mod.ColumnFeaturizer):
    column = "fake_idx"

    def compute(self, windows, records):
        return [[float(i), -float(i)] for i in range(len(windows))]


def _record(vid: str, text: str) -> VideoRecord:
    return VideoRecord(
        vid, vid, 1, "neutral", "train", None, None, 10.0, [], text, 1, [], [], [], {}
    )


@pytest.fixture
def cols_cfg(tmp_path, monkeypatch):
    windows = _windows()
    interim = tmp_path / "interim"
    save_window_index(windows, interim / "windows_index.parquet")
    pq = tmp_path / "feats.parquet"
    pl.DataFrame(
        {"id": [w.video_id for w in windows], "t0": [w.t0 for w in windows],
         "t1": [w.t1 for w in windows], "audio_emb": [[0.0]] * len(windows)},
        schema_overrides={"t0": pl.Float32, "t1": pl.Float32},
    ).write_parquet(pq)  # fmt: skip
    pq.with_suffix(".json").write_text(json.dumps({"feature_names": {}, "dims": {}}))
    monkeypatch.setitem(columns_mod.COLUMN_FEATURIZERS, "fake_len", lambda cfg: _VideoLen())
    monkeypatch.setitem(columns_mod.COLUMN_FEATURIZERS, "fake_idx", lambda cfg: _WindowIdx())
    records = {"v1": _record("v1", "abc"), "v2": _record("v2", "abcdef")}
    monkeypatch.setattr("src.data.indexing.build_video_index", lambda cfg: list(records.values()))
    cfg = OmegaConf.create(
        {"columns": ["fake_len", "fake_idx"],
         "data": {"force_columns": False,
                  "paths": {"parquet_path": str(pq), "interim_dir": str(interim)}}}
    )  # fmt: skip
    return cfg, pq


def test_columns_are_joined_by_window_and_cached(cols_cfg):
    cfg, pq = cols_cfg
    summary = run_featurize_columns(cfg)
    assert summary["columns"] == ["fake_len", "fake_idx"]
    df = pl.read_parquet(pq).sort(["id", "t0"])
    assert df["fake_len"].to_list() == [[3.0]] * 3 + [[6.0]] * 2  # broadcast por vídeo
    assert df["fake_idx"].to_list()[1] == [1.0, -1.0]
    sidecar = json.loads(pq.with_suffix(".json").read_text())
    assert sidecar["feature_names"]["fake_len"] == ["fake_len_0"]
    assert sidecar["dims"]["d_fake_len"] == 1
    assert not list(pq.parent.glob("*.tmp.parquet"))
    assert run_featurize_columns(cfg).get("cached") is True  # já existem → não recomputa


def test_merge_zero_fills_windows_missing_from_index(tmp_path):
    windows = _windows()
    pq = tmp_path / "f.parquet"
    pl.DataFrame(
        {"id": [w.video_id for w in windows], "t0": [w.t0 for w in windows],
         "t1": [w.t1 for w in windows]},
        schema_overrides={"t0": pl.Float32, "t1": pl.Float32},
    ).write_parquet(pq)  # fmt: skip
    kept = windows[:3]  # v2 fora do índice
    merged = merge_window_columns(pq, kept, {"vec": [[1.0, 2.0]] * 3, "txt": ["a"] * 3})
    v2 = merged.filter(pl.col("id") == "v2")
    assert v2["vec"].to_list() == [[0.0, 0.0]] * 2 and v2["txt"].to_list() == ["", ""]
    assert np.allclose(np.vstack(merged.filter(pl.col("id") == "v1")["vec"].to_numpy()), [1, 2])


def test_unknown_column_is_rejected():
    with pytest.raises(KeyError, match="desconhecida"):
        columns_mod.build_column_featurizer("nope", None)


# ---- cena: VideoMAE-v2 (código remoto: entrada (B, C, T, H, W), saída já é o vetor) ----


def test_scene_embedder_supports_remote_videomae_v2_layout(monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    import src.features.vision_embedder as ve

    seen: list[tuple] = []

    class _Processor:
        def __call__(self, frames, return_tensors="pt"):
            return {"pixel_values": torch.zeros(1, len(frames), 3, 8, 8)}  # (B, T, C, H, W)

    class _RemoteModel:
        config = SimpleNamespace()  # sem hidden_size → dimensão sai de 1 forward

        def __call__(self, pixel_values):
            seen.append(tuple(pixel_values.shape))
            return torch.ones(pixel_values.shape[0], 5)  # tensor direto, sem last_hidden_state

    monkeypatch.setattr(ve, "_load_hf_vision", lambda *a, **k: (_Processor(), _RemoteModel()))
    monkeypatch.setattr(ve, "read_uniform_frames", lambda path, n: np.zeros((n, 8, 8, 3)))
    emb = ve.SceneEmbedder("remote", num_frames=4, input_layout="bcthw", trust_remote_code=True,
                           device="cpu")  # fmt: skip
    out = emb.extract(["a.mp4", "b.mp4"])
    assert emb.dim == 5 and out.shape == (2, 5) and np.allclose(out, 1.0)
    assert seen[-1] == (1, 3, 4, 8, 8)  # (B, C, T, H, W)
    with pytest.raises(ValueError, match="input_layout"):
        ve.SceneEmbedder("remote", input_layout="xyz", device="cpu")


def test_parallel_merges_of_different_columns_keep_both(tmp_path):
    """Duas colunas gravadas ao mesmo tempo no mesmo Parquet (ex.: áudio numa GPU, rosto na
    outra): a trava faz cada merge reler o arquivo, então nenhuma sobrescreve a outra."""
    from concurrent.futures import ThreadPoolExecutor

    windows = _windows()
    pq = tmp_path / "f.parquet"
    pl.DataFrame(
        {"id": [w.video_id for w in windows], "t0": [w.t0 for w in windows],
         "t1": [w.t1 for w in windows]},
        schema_overrides={"t0": pl.Float32, "t1": pl.Float32},
    ).write_parquet(pq)  # fmt: skip
    cols = {f"c{i}": [[float(i)]] * len(windows) for i in range(4)}
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda c: merge_window_columns(pq, windows, {c: cols[c]}), cols))
    assert set(cols) <= set(pl.read_parquet(pq).columns)
