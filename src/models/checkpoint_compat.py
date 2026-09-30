"""Compatibilidade de checkpoint: snapshot/restauração de hiperparâmetros de arquitetura."""

from __future__ import annotations

from typing import Any

from src.logger import get_logger

log = get_logger("models.checkpoint_compat")

_ARCH_ATTRS = (
    "dim_a",
    "dim_b",
    "dim_tab",
    "hidden_channels",
    "heads",
    "out_channels",
    "common_dim",
    "gcn_out_dim",
    "top_k",
    "fusion",
    "use_tabular",
    "num_heads",
    "dropout",
    "use_latent_gcn",
    "use_bilstm",
    "lstm_hidden",
    "gat_num_layers",
    "lstm_num_layers",
    "use_tab_enhanced",
    "tab_pool",
    "use_ca_edge_weights",
    "ca_num_heads",
    "pool",
    "tab_fusion",
    "contrastive",
)


def snapshot_model_cfg(model: Any) -> dict[str, Any]:
    """Captura hiperparâmetros de arquitetura do objeto fusion."""
    cfg: dict[str, Any] = {}
    for key in _ARCH_ATTRS:
        if hasattr(model, key):
            val = getattr(model, key)
            if val is not None:
                cfg[key] = val
    return cfg


def restore_model_cfg(model: Any, cfg: dict[str, Any]) -> None:
    """Sobrepõe atributos de arquitetura no objeto fusion."""
    for key, val in cfg.items():
        if hasattr(model, key):
            setattr(model, key, val)


def infer_model_cfg_from_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Infere hiperparâmetros a partir dos shapes do ``state_dict`` (fallback legado)."""
    cfg: dict[str, Any] = {}

    audio_w = state_dict.get("model.gat.projections.audio.weight")
    hetero_proj_a = state_dict.get("model.proj_a.weight")
    if hetero_proj_a is not None:
        cfg["common_dim"] = int(hetero_proj_a.shape[0])
        cfg["dim_a"] = int(hetero_proj_a.shape[1])
        hetero_proj_b = state_dict.get("model.proj_b.weight")
        if hetero_proj_b is not None:
            cfg["dim_b"] = int(hetero_proj_b.shape[1])
    if audio_w is not None:
        cfg["hidden_channels"] = int(audio_w.shape[0])
        att = state_dict.get("model.gat.conv1.convs.<audio___temporal___audio>.att_src")
        if att is not None:
            cfg["heads"] = int(att.shape[1])
        out_w = state_dict.get("model.gat.conv2.convs.<audio___temporal___audio>.lin.weight")
        if out_w is not None:
            cfg["out_channels"] = int(out_w.shape[0])
        refine_w = state_dict.get("model.refine.0.weight")
        if refine_w is not None:
            in_dim = int(refine_w.shape[1])
            out_dim = int(refine_w.shape[0])
            cfg["out_channels"] = out_dim
            if state_dict.get("model.tab_encoder.proj_tab.weight") is not None:
                cfg["use_tab_enhanced"] = True
            elif in_dim == out_dim * 2:
                cfg["use_tab_enhanced"] = True
            else:
                cfg["use_tab_enhanced"] = False
        classifier_w = state_dict.get("model.classifier.0.weight")
        if classifier_w is not None and "out_channels" not in cfg:
            # multimodal_hetero_full: readout concat; out_channels ≈ GAT branch width
            gat_out = state_dict.get("model.gat.conv2.convs.<audio___temporal___audio>.lin.weight")
            if gat_out is not None:
                cfg["out_channels"] = int(gat_out.shape[0])
        fused_w = state_dict.get("model.gat.projections.fused.weight")
        if fused_w is not None:
            cfg["common_dim"] = int(fused_w.shape[1])
        log.info(f"Arquitetura inferida do ckpt: {cfg}")
        return cfg

    proj_a = state_dict.get("model.proj_a.weight")
    if proj_a is not None:
        cfg["common_dim"] = int(proj_a.shape[0])
        gcn_w = state_dict.get("model.gcn.gcn_weight.weight")
        if gcn_w is not None:
            cfg["gcn_out_dim"] = int(gcn_w.shape[0])
        clf_w = state_dict.get("model.classifier.0.weight")
        if clf_w is not None:
            cfg["gcn_out_dim"] = int(clf_w.shape[1]) if "gcn_out_dim" not in cfg else cfg["gcn_out_dim"]
        log.info(f"Arquitetura inferida do ckpt (baseline): {cfg}")
        return cfg

    return cfg


def load_state_dict_from_ckpt(ckpt_path: str) -> dict[str, Any]:
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return dict(ckpt["state_dict"])
    if isinstance(ckpt, dict):
        return dict(ckpt)
    return {}


def resolve_model_cfg_for_load(
    model: Any,
    trainer_state: dict[str, Any],
    ckpt_path: str | None,
) -> None:
    """Restaura arquitetura do modelo para casar com o checkpoint."""
    cfg: dict[str, Any] = dict(trainer_state.get("model_cfg") or {})
    if ckpt_path:
        inferred = infer_model_cfg_from_state_dict(load_state_dict_from_ckpt(ckpt_path))
        # Completa chaves ausentes (runs antigos / Optuna sem use_tab_enhanced no JSON).
        for key, val in inferred.items():
            if key not in cfg:
                cfg[key] = val
    if cfg:
        restore_model_cfg(model, cfg)
        log.info(f"Arquitetura restaurada: {cfg}")
        return

    if ckpt_path:
        inferred = infer_model_cfg_from_state_dict(load_state_dict_from_ckpt(ckpt_path))
        if inferred:
            restore_model_cfg(model, inferred)
            return

    log.warning(
        "Checkpoint sem model_cfg e shapes não reconhecidos — "
        "usando config YAML (pode causar size mismatch)."
    )
