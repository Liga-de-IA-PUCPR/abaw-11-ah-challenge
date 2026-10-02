"""Modos de fusão das features de suporte nos modelos com vídeo do Rodrigo (ablação).

Cobre ``model.tab_fusion`` do ``multimodal_hetero_face`` (graph | late | token | none) e do
``face_gnn_ts`` (mean | late), o controle ``face.enabled=false`` e a cópia da coluna facial
entre caches (``data.face_from``). Modelos minúsculos, dados sintéticos, sem checkpoints.
"""

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("torch_geometric")
pytest.importorskip("gnn_modalblocks")

from src.features.face_mesh import LANDMARK_DIM, NUM_FACE_LANDMARKS  # noqa: E402

B, T, D_A, D_B, D_TAB = 3, 4, 6, 5, 4
FACE = {
    "spatial_hidden": 4,
    "spatial_out": 4,
    "temporal_hidden": 4,
    "temporal_out": 4,
    "top_k": 2,
    "landmark_stride": 32,
    "temporal_mode": "gnn4ts",
}


def _batch(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor([4, 3, 2])
    mask = torch.arange(T).unsqueeze(0) >= lengths.unsqueeze(1)
    return {
        "audio_seq": torch.randn(B, T, D_A, generator=g),
        "text_seq": torch.randn(B, T, D_B, generator=g),
        "tab_seq": torch.randn(B, T, D_TAB, generator=g) * 10 + 3,
        "face_seq": torch.rand(B, T, NUM_FACE_LANDMARKS, LANDMARK_DIM, generator=g),
        "lengths": lengths,
        "key_padding_mask": mask,
        "label": torch.tensor([[1.0], [0.0], [1.0]]),
        "video_id": ["a", "b", "c"],
    }


def _multimodal(tab_fusion: str, face: bool = True, audio_norm: bool = False):
    from src.models.multimodal_hetero_face import MultimodalHeteroFaceFusion

    cfg = OmegaConf.create(
        {
            "name": "multimodal_hetero_face",
            "dim_a": D_A,
            "dim_b": D_B,
            "dim_tab": D_TAB,
            "common_dim": 8,
            "hidden_channels": 8,
            "heads": 2,
            "out_channels": 4,
            "gcn_out_dim": 4,
            "top_k": 2,
            "lstm_hidden": 4,
            "tab_fusion": tab_fusion,
            "audio_norm": audio_norm,
            "contrastive": {"enabled": False},
            "face": {"enabled": face, **FACE},
        }
    )
    torch.manual_seed(0)
    return MultimodalHeteroFaceFusion(cfg).build_lightning_module()


def _face_only(tab_fusion: str = "mean", face: bool = True, use_tabular: bool = True):
    from src.models.face_gnn_ts_model import FaceGnnTsFusion

    cfg = OmegaConf.create(
        {
            "name": "face_gnn_ts",
            "dim_tab": D_TAB,
            "use_tabular": use_tabular,
            "tab_hidden": 4,
            "tab_fusion": tab_fusion,
            "face": {"enabled": face, **FACE},
        }
    )
    torch.manual_seed(0)
    return FaceGnnTsFusion(cfg).build_lightning_module()


def _proba(lit, batch) -> torch.Tensor:
    lit.eval()
    with torch.no_grad():
        return lit._shared_step(batch, log_aux=False)[1]


@pytest.mark.parametrize("tab_fusion", ["graph", "late", "token", "none"])
@pytest.mark.parametrize("face", [True, False])
def test_multimodal_tab_fusion_trains(tab_fusion, face):
    lit = _multimodal(tab_fusion, face=face)
    lit.train()
    loss, proba, _ = lit._shared_step(_batch(), log_aux=False)
    assert proba.shape == (B, 1) and torch.isfinite(loss)
    loss.backward()
    if tab_fusion in ("late", "token"):
        grads = [p.grad for n, p in lit.named_parameters() if "tab_encoder" in n]
        assert grads and all(g is not None for g in grads)


def test_multimodal_graph_keeps_original_checkpoint_layout():
    keys = set(_multimodal("graph").model.state_dict())
    new = {"tab_encoder", "token_fuse", "token_norm", "audio_norm"}
    assert not any(part in new for k in keys for part in k.split("."))
    assert any(".face_encoder." in f".{k}" for k in keys)


def test_multimodal_audio_norm_standardizes_raw_librosa_scale():
    """librosa cru (centroide/rolloff ~10³) vira feature de nó do GAT; o BN o padroniza."""
    lit = _multimodal("token", audio_norm=True)
    b = _batch()
    b_raw = {**b, "audio_seq": b["audio_seq"] * 1000 + 2000}
    lit.train()
    loss, _, _ = lit._shared_step(b_raw, log_aux=False)
    assert torch.isfinite(loss)
    assert any("audio_norm" in k for k in lit.model.state_dict())


def test_masked_batch_norm_is_scale_free_and_skips_padding():
    from src.models.tab_fusion import masked_batch_norm

    b = _batch()
    x, mask = b["audio_seq"], b["key_padding_mask"]
    bn = torch.nn.BatchNorm1d(D_A, affine=False)
    out = masked_batch_norm(bn, x, mask)
    out_raw = masked_batch_norm(bn, x * 1000 + 2000, mask)
    torch.testing.assert_close(out[~mask], out_raw[~mask], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out[mask], x[mask])  # padding intacto


@pytest.mark.parametrize("tab_fusion", ["late", "token", "graph"])
def test_multimodal_tab_reaches_the_output(tab_fusion):
    lit, b = _multimodal(tab_fusion), _batch()
    b2 = {**b, "tab_seq": b["tab_seq"] + 5.0}
    assert not torch.allclose(_proba(lit, b), _proba(lit, b2))


def test_multimodal_none_ignores_tab():
    lit, b = _multimodal("none"), _batch()
    b2 = {**b, "tab_seq": b["tab_seq"] + 5.0}
    torch.testing.assert_close(_proba(lit, b), _proba(lit, b2))


def test_multimodal_rejects_unknown_tab_fusion():
    with pytest.raises(ValueError, match="tab_fusion"):
        _multimodal("concat")


@pytest.mark.parametrize("tab_fusion", ["mean", "late"])
def test_face_only_tab_fusion_trains(tab_fusion):
    lit = _face_only(tab_fusion)
    lit.train()
    loss, proba, _ = lit._shared_step(_batch(), log_aux=False)
    assert proba.shape == (B, 1) and torch.isfinite(loss)
    loss.backward()


def test_face_only_without_face_is_tab_only_and_needs_no_landmarks():
    lit = _face_only("late", face=False)
    b = {k: v for k, v in _batch().items() if k != "face_seq"}
    assert not any("face_encoder" in n for n, _ in lit.named_parameters())
    assert _proba(lit, b).shape == (B, 1)


def test_face_only_rejects_empty_model():
    with pytest.raises(ValueError, match="use_tabular"):
        _face_only(face=False, use_tabular=False)


def test_featurize_face_copies_column_from_other_cache(tmp_path):
    from src.pipeline.featurize_face import run_featurize_face

    keys = {
        "id": ["v1", "v1", "v2"],
        "t0": [0.0, 2.5, 0.0],
        "t1": [5.0, 7.5, 5.0],
    }
    d_face = NUM_FACE_LANDMARKS * LANDMARK_DIM
    src = tmp_path / "librosa_face.parquet"  # v2 sem rosto: ausente na origem
    pl.DataFrame(
        {
            **{k: v[:2] for k, v in keys.items()},
            "audio_emb": [[1.0]] * 2,
            "face_landmarks": [[float(i)] * d_face for i in range(2)],
        },
        schema_overrides={"t0": pl.Float32, "t1": pl.Float32},
    ).write_parquet(src)
    dst = tmp_path / "w2v_face.parquet"
    pl.DataFrame(
        {**keys, "audio_emb": [[2.0, 2.0]] * 3},
        schema_overrides={"t0": pl.Float32, "t1": pl.Float32},
    ).write_parquet(dst)

    cfg = OmegaConf.create({"data": {"face_from": str(src), "paths": {"parquet_path": str(dst)}}})
    summary = run_featurize_face(cfg)
    assert summary["face_from"] == str(src) and summary["d_face"] == d_face

    df = pl.read_parquet(dst).sort(["id", "t0"])
    face = np.vstack(df["face_landmarks"].to_numpy())
    assert df["audio_emb"].to_list() == [[2.0, 2.0]] * 3  # o cache de destino é preservado
    assert np.allclose(face[:2, 0], [0.0, 1.0])
    assert np.count_nonzero(face[2]) == 0  # janela ausente na origem → zeros
