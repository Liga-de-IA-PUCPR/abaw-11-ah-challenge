"""Testes do modelo MoE: blocos, ramos, fusão, dataset por colunas e treino ponta a ponta."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from src.models.blocks import (  # noqa: E402
    GatedAttentionPool,
    MaskedBatchNorm,
    SoftMoE,
    StatsPool,
    TemporalEncoder,
    load_balance_loss,
    masked_max,
    masked_mean,
)
from src.models.moe_fusion import MoEFusion, TextAnchoredMoE  # noqa: E402

MASK = torch.tensor([[False, False, True], [False, False, False]])  # vídeo 0 tem T=2


def _seq(b=2, t=3, d=4):
    torch.manual_seed(0)
    return torch.randn(b, t, d)


# ---- blocos ------------------------------------------------------------------


def test_masked_pools_ignore_padding():
    x = _seq()
    poisoned = x.clone()
    poisoned[0, 2] = 1e6  # janela de padding não pode vazar
    for fn in (masked_mean, masked_max):
        torch.testing.assert_close(fn(x, MASK), fn(poisoned, MASK))
    torch.testing.assert_close(masked_mean(x, MASK)[0], x[0, :2].mean(0))
    attn = GatedAttentionPool(4)
    torch.testing.assert_close(attn(x, MASK), attn(poisoned, MASK))


def test_stats_pool_is_mu_sigma_and_deltas():
    x = _seq()
    out = StatsPool()(x, MASK)
    assert out.shape == (2, 16)
    mu, sigma, dmu = out[0, :4], out[0, 4:8], out[0, 8:12]
    torch.testing.assert_close(mu, x[0, :2].mean(0))
    torch.testing.assert_close(sigma, x[0, :2].std(0, unbiased=False))
    torch.testing.assert_close(dmu, x[0, 1] - x[0, 0])  # 1 par válido


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_temporal_encoder_valid_steps_ignore_padding(kind):
    enc = TemporalEncoder(kind, dim=4, num_heads=2, dropout=0.0).eval()
    x = _seq()
    poisoned = x.clone()
    poisoned[0, 2] = 50.0
    torch.testing.assert_close(enc(x, MASK)[0, :2], enc(poisoned, MASK)[0, :2])


def test_soft_moe_weights_and_shapes():
    moe = SoftMoE(4, 6, num_experts=3)
    out, w = moe(_seq())
    assert out.shape == (2, 3, 6) and w.shape == (2, 3, 3)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 3))


def test_load_balance_only_above_threshold():
    balanced = torch.full((8, 4), 0.25)
    assert load_balance_loss(balanced).item() == 0.0
    skewed = torch.tensor([[0.9, 0.05, 0.05]] * 8)
    assert load_balance_loss(skewed, threshold=0.7).item() > 0.0
    assert load_balance_loss(skewed, threshold=0.95).item() == 0.0


def test_masked_batchnorm_single_row_uses_running_stats():
    bn = MaskedBatchNorm(3).train()
    one = torch.randn(1, 3)
    torch.testing.assert_close(bn(one), one / torch.sqrt(torch.tensor(1.0 + bn.bn.eps)))


# ---- modelo ------------------------------------------------------------------

BRANCHES = {
    "text": {"role": "anchor", "encoder": "sequence", "column": "text_emb", "dim": 8},
    "audio": {
        "role": "residual", "encoder": "sequence", "column": "audio_emb_x", "dim": 8,
        "temporal": "gru", "window_weight": 0.1,
    },
    "asr": {"role": "residual", "encoder": "vector", "column": "asr_timing", "dim": 4},
    "markers": {
        "role": "head", "encoder": "vector", "column": "hesitation_markers", "dim": 4,
        "aux_weight": 0.3,
    },
}  # fmt: skip
DIMS = {"text_emb": 6, "audio_emb_x": 4, "asr_timing": 5, "hesitation_markers": 3}


def _batch(b=2, t=3):
    torch.manual_seed(1)
    return {
        "features": {c: torch.randn(b, t, d) for c, d in DIMS.items()},
        "key_padding_mask": MASK,
        "label": torch.tensor([[1.0], [0.0]]),
        "window_label": torch.tensor([[1, 0, -1], [0, -1, 0]]),
        "video_id": ["a", "b"],
    }


def test_text_anchored_moe_forward():
    net = TextAnchoredMoE(BRANCHES, DIMS, dim=8, num_experts=3)
    out = net(_batch())
    assert out["logit"].shape == (2, 1) and out["embedding"].shape == (2, 8)
    assert set(out["gates"]) == {"audio", "asr"} and out["gates"]["audio"].shape == (2, 1)
    assert out["routing"]["head"].shape == (2, 3)
    assert out["aux_logits"]["markers"].shape == (2, 1)
    assert out["window_logits"]["audio"].shape == (2, 3)


def test_single_branch_is_the_unimodal_member():
    net = TextAnchoredMoE({"text": BRANCHES["text"]}, DIMS, dim=8, head="mlp")
    out = net(_batch())
    assert out["logit"].shape == (2, 1) and not out["gates"] and not out["routing"]


def test_exactly_one_anchor_required():
    with pytest.raises(ValueError, match="anchor"):
        TextAnchoredMoE({"a": BRANCHES["audio"]}, DIMS)


def test_lit_module_losses_include_aux_window_and_balance():
    from src.models.moe_fusion import LitMoEFusion

    net = TextAnchoredMoE(BRANCHES, DIMS, dim=8, num_experts=3)
    lit = LitMoEFusion(net, balance_weight=1.0, balance_threshold=0.0)
    parts = lit._losses(net(_batch()), _batch())
    assert {"main", "aux_markers", "window_audio", "balance_head"} <= set(parts)
    assert all(torch.isfinite(v) for v in parts.values())


def test_data_spec_lists_columns_and_transcript():
    spec = MoEFusion.data_spec(
        {"branches": {**BRANCHES, "text": {"role": "anchor", "encoder": "hf_text",
                                           "model_name": "m", "max_length": 64}}}
    )  # fmt: skip
    assert spec["columns"] == ["asr_timing", "audio_emb_x", "hesitation_markers"]
    assert spec["transcript"] == {"model_name": "m", "max_length": 64}


def test_hf_text_encoder_freezes_first_layers(monkeypatch):
    from transformers import AutoModel, RobertaConfig, RobertaModel

    from src.models.encoders import HFTextEncoder

    tiny = RobertaModel(RobertaConfig(vocab_size=50, hidden_size=16, num_hidden_layers=3,
                                      num_attention_heads=2, intermediate_size=32))  # fmt: skip
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *a, **k: tiny)
    enc = HFTextEncoder("tiny", num_frozen_layers=2)
    layers = enc.backbone.encoder.layer
    assert not any(p.requires_grad for p in layers[1].parameters())
    assert all(p.requires_grad for p in layers[2].parameters())
    h, _, _ = enc(torch.randint(3, 50, (2, 7)), torch.ones(2, 7, dtype=torch.long))
    assert h.shape == (2, 16)


# ---- dataset orientado a colunas ---------------------------------------------


def test_dataset_columns_window_labels_tokens_and_subset(window_parquet, monkeypatch):
    import src.data.datasets as ds_mod
    from src.data.datasets import VideoSequenceDataset, collate_sequences

    monkeypatch.setattr(
        ds_mod, "_tokenize", lambda texts, **_: ([list(range(1, 2 + len(t) % 5)) for t in texts], 0)
    )
    ds = VideoSequenceDataset(
        window_parquet, columns=["asr_timing", "tabular"], transcript={"model_name": "x"}
    )
    assert ds.dims["asr_timing"] == 5 and ds.dims["tabular"] == 3
    batch = collate_sequences([ds[0], ds[1]])
    t_max = batch["key_padding_mask"].shape[1]
    assert batch["features"]["asr_timing"].shape == (2, t_max, 5)
    assert batch["window_label"].shape == (2, t_max)
    assert (batch["window_label"][batch["key_padding_mask"]] == -1).all()
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    sub = ds.subset({ds.video_ids[1]})
    assert sub.video_ids == [ds.video_ids[1]] and len(sub) == 1
    torch.testing.assert_close(sub[0]["features"]["asr_timing"], ds[1]["features"]["asr_timing"])
    with pytest.raises(KeyError, match="featurize_columns"):
        VideoSequenceDataset(window_parquet, columns=["nao_existe"])


# ---- ponta a ponta: fit → save → load_trainer → embeddings ---------------------


def test_train_save_reload_exports_embeddings(compose_cfg):
    from src.data.datasets import as_loader, load_split
    from src.models.registry import create_model
    from src.training.factory import create_trainer, load_trainer

    cfg = compose_cfg(
        "+experiment=moe_r1_text_frozen",
        "+branch@model.branches.audio=audio_emotion",
        "model.branches.audio.column=audio_emb_x",
        "+branch@model.branches.asr=asr_timing",
        "model.fusion.head=moe",
        "model.fusion.dim=16",
        "trainer.max_epochs=2",
        "data.batch_size=8",
    )
    model, family = create_model(cfg.model.name, cfg.model)
    trainer = create_trainer(family, model=model, cfg=cfg)
    out_dir = f"{cfg.data.paths.output_root}/run"
    trainer.output_dir = out_dir
    trainer.fit(
        as_loader(cfg, load_split(cfg, "train", family=family), family, "train"),
        as_loader(cfg, load_split(cfg, "val", family=family), family, "val"),
    )
    trainer.save(out_dir)
    assert model.input_dims == {"asr_timing": 5, "audio_emb_x": 4, "hesitation_markers": 3,
                                "text_emb": 6}  # fmt: skip

    # Recarga só com o preset base: model_config salvo restaura ramos + dims.
    base = compose_cfg("+experiment=moe_r1_text_frozen")
    reloaded = load_trainer("lightning", out_dir, cfg=base)
    data = as_loader(
        reloaded.config,
        load_split(reloaded.config, "test", family="lightning"),
        "lightning",
        "test",
    )
    out = reloaded.video_outputs(data)
    assert out["embedding"].shape == (len(out["video_ids"]), 16)
    assert out["router_weights"].shape[1] == 4 and out["gates"].shape[1] == 2
    assert np.isfinite(out["y_proba"]).all()


# ---- vetor pré-logit de QUALQUER membro (CA/GNN não o devolvem no predict_step) ----


def test_prelogit_capture_picks_the_logit_layer():
    from torch import nn

    from src.training.lightning_trainer import _PrelogitCapture

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(3, 4)
            self.aux = nn.Linear(4, 1)  # isca: também (B, 4) → (B, 1), mas não gera a proba
            self.head = nn.Linear(4, 1)

        def forward(self, x):
            h = torch.relu(self.body(x))
            self.aux(h)
            return torch.sigmoid(self.head(h)).squeeze(1), h

    net = _Net().eval()
    capture = _PrelogitCapture(net)
    with torch.no_grad():
        outs = [net(torch.randn(5, 3)) for _ in range(2)]
    capture.remove()
    proba = torch.cat([p for p, _ in outs]).numpy()
    emb = capture.embedding_for(proba)
    np.testing.assert_allclose(emb, torch.cat([h for _, h in outs]).numpy(), rtol=1e-6)
    assert capture.embedding_for(proba + 0.3) is None  # nenhuma camada reproduz → sem embedding


def test_cross_attention_member_exports_prelogit_embeddings(compose_cfg):
    from scipy.special import expit

    from src.data.datasets import as_loader, load_split
    from src.models.registry import create_model
    from src.training.factory import create_trainer

    cfg = compose_cfg(
        "+experiment=cross_attention",
        "model.common_dim=8",
        "model.num_heads=2",
        "trainer.max_epochs=1",
        "data.batch_size=8",
    )
    model, family = create_model(cfg.model.name, cfg.model)
    trainer = create_trainer(family, model=model, cfg=cfg)
    trainer.output_dir = f"{cfg.data.paths.output_root}/ca"
    trainer.fit(
        as_loader(cfg, load_split(cfg, "train", family=family), family, "train"),
        as_loader(cfg, load_split(cfg, "val", family=family), family, "val"),
    )
    test = as_loader(cfg, load_split(cfg, "test", family=family), family, "test")
    out = trainer.predict_outputs(test)
    head = trainer._lit_module.model.classifier[-1]  # Linear(common_dim, 1) que gera o logit
    logit = out["embedding"] @ head.weight.detach().numpy().T + head.bias.detach().numpy()
    np.testing.assert_allclose(expit(logit[:, 0]), out["proba"], atol=1e-4)
    assert "embedding" not in trainer.predict_outputs(test, embeddings=False)
