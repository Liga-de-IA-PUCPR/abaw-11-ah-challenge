"""Testes do caminho de vídeo sem MediaPipe: featurize_face → Parquet → face_seq no batch.

O extrator real (MediaPipe + OpenCV sobre os .mp4) é trocado por um falso; o que se
testa é a integração com o pipeline da main: join por janela, janelas sem landmarks
preenchidas com zeros, escrita atômica e a sequência facial no ``VideoSequenceDataset``.
"""

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from src.data.schema import WindowSample
from src.data.windowing import save_window_index
from src.features.face_mesh import LANDMARK_DIM, NUM_FACE_LANDMARKS

D_FACE = NUM_FACE_LANDMARKS * LANDMARK_DIM


def _windows() -> list[WindowSample]:
    out = []
    for vid, split, n in [("v1", "train", 3), ("v2", "val", 2)]:
        for i in range(n):
            out.append(
                WindowSample(
                    window_id=f"{vid}#w{i}",
                    video_id=vid,
                    participant_id=vid,
                    t0=2.5 * i,
                    t1=2.5 * i + 5.0,
                    text="hmm well",
                    label=0,
                    question_type="neutral",
                    split=split,
                    video_label=int(vid == "v1"),
                )
            )
    return out


def _base_parquet(windows: list[WindowSample], path) -> None:
    n = len(windows)
    pl.DataFrame(
        {
            "id": [w.video_id for w in windows],
            "window_idx": np.arange(n, dtype=np.int32),
            "t0": [w.t0 for w in windows],
            "t1": [w.t1 for w in windows],
            "participant_id": [w.participant_id for w in windows],
            "question_type": [w.question_type for w in windows],
            "audio_emb": [[0.1] * 4] * n,
            "text_emb": [[0.2] * 6] * n,
            "tabular": [[0.3] * 2] * n,
            "label": [0] * n,
            "video_label": [w.video_label for w in windows],
            "split": [w.split for w in windows],
        },
        schema_overrides={"t0": pl.Float32, "t1": pl.Float32},
    ).write_parquet(path)


class _FakeExtractor:
    """Landmarks constantes = índice da janela dentro do vídeo (checável no Parquet)."""

    dim = D_FACE

    def __init__(self, **_):
        pass

    def extract(self, batch, video_root):
        shape = (NUM_FACE_LANDMARKS, LANDMARK_DIM)
        return [np.full(shape, i, dtype=np.float32) for i, _ in enumerate(batch)]

    def flatten(self, lms):
        return [lm.reshape(-1) for lm in lms]

    def close(self):
        pass


@pytest.fixture
def face_cfg(tmp_path):
    windows = _windows()
    interim = tmp_path / "interim"
    interim.mkdir()
    save_window_index(windows, interim / "windows_index.parquet")
    pq = tmp_path / "feats.parquet"
    _base_parquet(windows, pq)
    cfg = OmegaConf.create(
        {
            "data": {
                "force_face": False,
                "paths": {
                    "parquet_path": str(pq),
                    "interim_dir": str(interim),
                    "data_root": str(tmp_path),
                },
            }
        }
    )
    return cfg, pq


def test_featurize_face_adds_column_and_zero_fills(face_cfg, monkeypatch):
    import src.pipeline.featurize_face as ff

    cfg, pq = face_cfg

    class _NoFaceInV2(_FakeExtractor):
        """Simula "sem rosto" em v2: o extrator devolve landmarks zerados p/ essas janelas."""

        def extract(self, batch, video_root):
            lms = super().extract(batch, video_root)
            return [lm * 0 if w.video_id == "v2" else lm for lm, w in zip(lms, batch, strict=True)]

    monkeypatch.setattr(ff, "FaceMeshExtractor", _NoFaceInV2)
    summary = ff.run_featurize_face(cfg)

    assert summary["d_face"] == D_FACE
    df = pl.read_parquet(pq).sort(["id", "window_idx"])
    assert "face_landmarks" in df.columns
    assert df.height == 5  # nenhuma janela perdida no join
    v1 = np.vstack(df.filter(pl.col("id") == "v1")["face_landmarks"].to_numpy())
    v2 = np.vstack(df.filter(pl.col("id") == "v2")["face_landmarks"].to_numpy())
    assert v1.shape == (3, D_FACE) and np.allclose(v1[:, 0], [0, 1, 2])
    assert np.count_nonzero(v2) == 0  # sem landmarks → zeros
    assert not list(pq.parent.glob("*.tmp.parquet"))  # escrita atômica limpa o tmp

    # 2ª chamada: cache (não recomputa sem data.force_face)
    assert ff.run_featurize_face(cfg).get("cached") is True


def test_video_sequence_dataset_exposes_face_seq(face_cfg, monkeypatch):
    import src.pipeline.featurize_face as ff

    torch = pytest.importorskip("torch")
    from src.data.datasets import VideoSequenceDataset, collate_sequences

    cfg, pq = face_cfg
    monkeypatch.setattr(ff, "FaceMeshExtractor", _FakeExtractor)
    ff.run_featurize_face(cfg)

    ds = VideoSequenceDataset(pq)
    assert ds.has_face and ds.dim_face == (NUM_FACE_LANDMARKS, LANDMARK_DIM)
    batch = collate_sequences([ds[i] for i in range(len(ds))])
    assert batch["face_seq"].shape == (2, 3, NUM_FACE_LANDMARKS, LANDMARK_DIM)  # T_max=3
    assert batch["face_seq"].dtype == torch.float32


def test_dataset_without_face_column_has_no_face_seq(tmp_path):
    pytest.importorskip("torch")
    from src.data.datasets import VideoSequenceDataset

    pq = tmp_path / "feats.parquet"
    _base_parquet(_windows(), pq)
    ds = VideoSequenceDataset(pq)
    assert not ds.has_face and "face_seq" not in ds[0]
