"""Testes do fine-tune de texto GoEmotions (FASE 1) — sem baixar o modelo real da HF Hub.

Usa um ``RobertaModel`` tiny com pesos aleatórios (mesma arquitetura do
``SamLowe/roberta-base-go_emotions``, mas ``RobertaConfig`` local — nenhuma
chamada de rede) para validar: shapes do forward, finitude/diferenciabilidade
da loss, o termo de consistência R-Drop (zero quando os dois passes são
idênticos, ex.: ``eval()``) e a cabeça auxiliar de hesitação (74-d).
"""

from __future__ import annotations

import torch
from transformers import RobertaConfig, RobertaModel

from src.models.text_finetune import (
    TextFinetuneNet,
    freeze_backbone_layers,
    symmetric_kl_from_logits,
)

HIDDEN_SIZE = 32
DIM_TAB = 74


def _tiny_backbone() -> RobertaModel:
    """RobertaModel tiny/random-init — mesma arquitetura, zero downloads."""
    config = RobertaConfig(
        vocab_size=99,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=66,
        type_vocab_size=1,
        pad_token_id=1,
    )
    return RobertaModel(config)


def _build_module(dropout: float = 0.3, dim_tab: int = DIM_TAB):
    backbone = _tiny_backbone()
    freeze_backbone_layers(backbone, num_frozen_layers=1)
    module = TextFinetuneNet(
        backbone=backbone,
        hidden_size=HIDDEN_SIZE,
        dropout=dropout,
        head_hidden=16,
        dim_tab=dim_tab,
    )
    return module


def _dummy_batch(batch_size: int = 4, seq_len: int = 10, dim_tab: int = DIM_TAB):
    input_ids = torch.randint(low=2, high=99, size=(batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    tab = torch.randn(batch_size, dim_tab)
    label = torch.randint(0, 2, size=(batch_size, 1)).float()
    return input_ids, attention_mask, tab, label


# =============================================================================
# Congelamento de camadas
# =============================================================================
def test_freeze_backbone_layers_freezes_embeddings_and_first_n():
    backbone = _tiny_backbone()
    freeze_backbone_layers(backbone, num_frozen_layers=1)

    assert all(not p.requires_grad for p in backbone.embeddings.parameters())
    assert all(not p.requires_grad for p in backbone.encoder.layer[0].parameters())
    # camadas restantes (>= num_frozen_layers) permanecem treináveis
    assert any(p.requires_grad for p in backbone.encoder.layer[1].parameters())
    assert any(p.requires_grad for p in backbone.encoder.layer[2].parameters())


# =============================================================================
# Forward: shapes
# =============================================================================
def test_forward_shapes():
    module = _build_module()
    input_ids, attention_mask, tab, _ = _dummy_batch(batch_size=5, seq_len=12)

    logit, aux_logit = module(input_ids, attention_mask, tab=tab)

    assert logit.shape == (5, 1)
    assert aux_logit.shape == (5, 1)


def test_forward_without_tab_has_no_aux_logit():
    module = _build_module()
    input_ids, attention_mask, _, _ = _dummy_batch(batch_size=3)

    logit, aux_logit = module(input_ids, attention_mask, tab=None)

    assert logit.shape == (3, 1)
    assert aux_logit is None


# =============================================================================
# Loss: finita e diferenciável
# =============================================================================
def test_main_loss_is_finite_and_backward_works():
    module = _build_module()
    input_ids, attention_mask, tab, label = _dummy_batch(batch_size=4)

    logit, aux_logit = module(input_ids, attention_mask, tab=tab)
    main_loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label)
    aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux_logit, label)
    loss = main_loss + 0.3 * aux_loss

    assert torch.isfinite(loss)
    loss.backward()

    # gradiente chegou na head principal (treinável) e na aux head
    head_grad_norms = [p.grad.norm().item() for p in module.head.parameters() if p.grad is not None]
    assert len(head_grad_norms) > 0
    assert all(g >= 0 for g in head_grad_norms)
    assert module.aux_head.weight.grad is not None
    assert torch.isfinite(module.aux_head.weight.grad).all()

    # camada congelada (índice 0) não recebeu gradiente (requires_grad=False)
    frozen_layer = module.backbone.encoder.layer[0]
    assert all(p.grad is None for p in frozen_layer.parameters())


# =============================================================================
# R-Drop: KL simétrico
# =============================================================================
def test_rdrop_kl_zero_when_passes_are_identical():
    logit = torch.randn(6, 1)
    kl = symmetric_kl_from_logits(logit, logit.clone())
    assert torch.allclose(kl, torch.zeros(()), atol=1e-6)


def test_rdrop_kl_positive_when_passes_differ():
    torch.manual_seed(0)
    logit1 = torch.randn(6, 1)
    logit2 = torch.randn(6, 1)
    kl = symmetric_kl_from_logits(logit1, logit2)
    assert kl.item() > 0


def test_rdrop_kl_zero_in_eval_mode_two_forward_passes():
    """Em eval() o dropout é desligado -> dois forward passes do MESMO batch
    devem produzir logits idênticos, e o termo de consistência R-Drop deve ser 0."""
    module = _build_module(dropout=0.5)
    module.eval()
    input_ids, attention_mask, tab, _ = _dummy_batch(batch_size=4)

    with torch.no_grad():
        logit1, _ = module(input_ids, attention_mask, tab=tab)
        logit2, _ = module(input_ids, attention_mask, tab=tab)

    kl = symmetric_kl_from_logits(logit1, logit2)
    assert torch.allclose(logit1, logit2, atol=1e-6)
    assert torch.allclose(kl, torch.zeros(()), atol=1e-6)


# =============================================================================
# Cabeça auxiliar (hesitação 74-d)
# =============================================================================
def test_aux_head_loss_changes_with_different_tab_inputs():
    module = _build_module()
    module.eval()
    input_ids, attention_mask, _, label = _dummy_batch(batch_size=4)

    tab_a = torch.zeros(4, DIM_TAB)
    tab_b = torch.ones(4, DIM_TAB) * 5.0

    with torch.no_grad():
        _, aux_logit_a = module(input_ids, attention_mask, tab=tab_a)
        _, aux_logit_b = module(input_ids, attention_mask, tab=tab_b)

    loss_a = torch.nn.functional.binary_cross_entropy_with_logits(aux_logit_a, label)
    loss_b = torch.nn.functional.binary_cross_entropy_with_logits(aux_logit_b, label)

    assert not torch.allclose(aux_logit_a, aux_logit_b)
    assert not torch.allclose(loss_a, loss_b)


def test_no_aux_head_when_dim_tab_zero():
    module = _build_module(dim_tab=0)
    assert module.aux_head is None
