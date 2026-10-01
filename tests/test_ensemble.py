"""Testes do EnsembleTrainer heterogêneo (membros falsos, sem Lightning/checkpoints)."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lightning")

from src.features.builder import _video_aligned_spans  # noqa: E402
from src.training.ensemble import EnsembleMember, EnsembleTrainer  # noqa: E402


class _FakeLightning:
    """Membro lightning falso: devolve probas fixas (em ordem embaralhada)."""

    def __init__(self, probs: dict[str, float], threshold: float = 0.5):
        self.probs = probs
        self.threshold_ = threshold

    def _infer(self, loader):
        ids = list(reversed(list(self.probs)))  # ordem diferente do outro membro
        return np.asarray(ids), np.asarray([self.probs[v] for v in ids], dtype=np.float32)


class _FakeSklearn:
    def __init__(self, probs: dict[str, float]):
        self.probs = probs
        self.threshold_ = 0.4

    def predict_scores(self, view):
        return dict(self.probs)


def _loader(video_ids):
    ds = SimpleNamespace(video_ids=list(video_ids), parquet_path="unused.parquet")
    return SimpleNamespace(dataset=ds, batch_size=2, num_workers=0)


IDS = ["a", "b", "c"]


def test_unweighted_mean_matches_plain_average():
    m1 = _FakeLightning({"a": 0.2, "b": 0.8, "c": 0.5})
    m2 = _FakeLightning({"a": 0.4, "b": 0.6, "c": 0.1})
    ens = EnsembleTrainer([m1, m2], config=SimpleNamespace())  # trainers crus (compat.)
    ids, proba = ens._infer(_loader(IDS))
    assert list(ids) == IDS
    np.testing.assert_allclose(proba, [0.3, 0.7, 0.3], rtol=1e-6)
    assert ens.threshold_ == pytest.approx(0.5)  # média dos limiares antes de recalibrar


def test_weighted_mean_and_sklearn_member(monkeypatch):
    import src.data.datasets as datasets

    monkeypatch.setattr(datasets, "WindowMatrixView", lambda pq, ids: SimpleNamespace(ids=ids))
    lit = EnsembleMember(_FakeLightning({"a": 0.0, "b": 1.0, "c": 0.5}), name="gnn", weight=3.0)
    skl = EnsembleMember(
        _FakeSklearn({"a": 1.0, "b": 0.0, "c": 0.5}), name="cb", family="sklearn", weight=1.0
    )
    ens = EnsembleTrainer([lit, skl], config=SimpleNamespace())
    ids, proba = ens._infer(_loader(IDS))
    np.testing.assert_allclose(proba, [0.25, 0.75, 0.5], rtol=1e-6)


def test_members_must_cover_same_videos():
    m1 = _FakeLightning({"a": 0.2, "b": 0.8})
    m2 = _FakeLightning({"a": 0.4, "c": 0.6})
    ens = EnsembleTrainer([m1, m2], config=SimpleNamespace())
    with pytest.raises(ValueError, match="MESMO split"):
        ens._infer(_loader(["a", "b"]))


def test_video_aligned_spans_never_split_a_video():
    w = [SimpleNamespace(video_id=v) for v in "aaabbccccdde"]
    assert _video_aligned_spans(w, None) == [(0, 12)]
    spans = _video_aligned_spans(w, 3)
    assert spans == [(0, 3), (3, 9), (9, 12)]
    for a, b in spans:  # cada fatia começa no 1º e termina no último janela de um vídeo
        assert a == 0 or w[a].video_id != w[a - 1].video_id
        assert b == len(w) or w[b].video_id != w[b - 1].video_id
