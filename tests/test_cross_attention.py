"""Cross-attention do artigo + colunas extras opt-in (ex.: vídeo do plano MoE)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightning")

from src.models.cross_attention import CrossAttentionFusion, _build_fusion_module  # noqa: E402

PAPER = dict(
    dim_a=6, dim_b=4, common_dim=8, num_heads=2, dropout=0.3, dim_tab=3, use_tabular=True,
    pool="attention", tab_fusion="token",
)  # fmt: skip


def test_without_extra_columns_the_paper_model_has_no_new_parameters():
    keys = set(_build_fusion_module(**PAPER).state_dict())
    assert not any(k.startswith("extra") for k in keys)
    assert CrossAttentionFusion.data_spec({"extra_columns": []}) == {}


def test_extra_column_is_fused_trained_saved_and_reloaded(compose_cfg):
    from src.data.datasets import as_loader, load_split
    from src.models.registry import create_model
    from src.training.factory import create_trainer, load_trainer

    overrides = ("+experiment=cross_attention", "model.common_dim=8", "model.num_heads=2")
    cfg = compose_cfg(
        *overrides, "model.extra_columns=[audio_emb_x]", "trainer.max_epochs=1",
        "data.batch_size=8",
    )  # fmt: skip
    model, family = create_model(cfg.model.name, cfg.model)
    trainer = create_trainer(family, model=model, cfg=cfg)
    out_dir = f"{cfg.data.paths.output_root}/ca_video"
    trainer.output_dir = out_dir
    trainer.fit(
        as_loader(cfg, load_split(cfg, "train", family=family), family, "train"),
        as_loader(cfg, load_split(cfg, "val", family=family), family, "val"),
    )
    trainer.save(out_dir)
    assert model.input_dims == {"audio_emb_x": 4}
    keys = trainer._lit_module.state_dict()
    assert "model.extra.audio_emb_x.1.weight" in keys

    # Recarga com o preset do artigo na CLI: o model_config salvo traz a coluna extra.
    reloaded = load_trainer("lightning", out_dir, cfg=compose_cfg(*overrides))
    data = as_loader(
        reloaded.config, load_split(reloaded.config, "test", family="lightning"), "lightning", "x"
    )
    out = reloaded.video_outputs(data)
    assert np.isfinite(out["y_proba"]).all() and out["embedding"].shape[1] == 8
