"""CrossAttentionFusion + LitCrossAttention — modelo de competição (família lightning).

UPGRADE do ``model_cross_attention.py`` da branch ``matheus`` (que recebia UMA
janela por vídeo, T=1) para a **sequência de janelas** do vídeo (T = nº de janelas
> 1, README §6.2), fazendo modelagem temporal real.

Arquitetura (sobre tensores ``(B, T, D)``):
  proj_a/proj_b -> common_dim
  MultiheadAttention(q=áudio, kv=texto, key_padding_mask=máscara de padding)
  residual + LayerNorm
  POOLING TEMPORAL MASCARADO sobre T (média ignorando janelas de padding)
  MLP -> 1 logit por vídeo (BCEWithLogits).

⚠️ **Imports lazy:** ``torch``/``lightning``/``torchmetrics`` são importados DENTRO
das funções/métodos. O módulo só é importado pelo registry quando
``model=cross_attention`` — mantendo o caminho RandomForest livre dessas deps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.logger import get_logger

if TYPE_CHECKING:  # só p/ type-checkers; não executa em runtime
    import torch

log = get_logger("models.cross_attention")


# =============================================================================
# Construtor lazy do nn.Module de fusão
# =============================================================================
def _build_fusion_module(
    dim_a: int,
    dim_b: int,
    common_dim: int,
    num_heads: int,
    dropout: float,
    dim_tab: int = 0,
    use_tabular: bool = False,
    pool: str = "mean",
    tab_fusion: str = "late",
):
    """Constrói o ``nn.Module`` de cross-attention (importa torch *lazy*).

    Definido como factory para que ``import torch`` só ocorra quando a
    cross-attention é efetivamente instanciada (não no import do módulo).

    Args:
        dim_a / dim_b: dims dos embeddings de áudio / texto por janela.
        common_dim / num_heads / dropout: hiperparâmetros da fusão.
        dim_tab: dim do vetor tabular por janela (features de hesitação + tabulares).
        use_tabular: se ``True`` (e ``dim_tab > 0``), funde o ``tab_seq`` (features de
            suporte: hesitação acústica + ambivalência textual).
        pool: **agregação temporal janela→vídeo** sobre T. ``"mean"`` (default legado):
            média mascarada — trata A/H como uniforme no vídeo, DILUI eventos localizados.
            ``"attention"``: pooling atenção-MIL gated (Ilse et al. 2018) — aprende um peso
            por janela e faz soma ponderada, deixando as janelas informativas dominarem
            (o alvo é MIL: vídeo positivo se ALGUMA janela tem A/H). ``"max"``: máximo
            mascarado (a janela mais forte manda; barato, sem parâmetros).
        tab_fusion: COMO o ``tab_seq`` entra. ``"late"`` (default legado): mean-pool do tab
            → proj → concat DEPOIS do pooling temporal (ramo paralelo, não passa pela
            atenção nem informa o pooling). ``"token"``: projeta o tab de CADA janela e o
            soma ao token daquela janela ANTES do pooling — assim as features de suporte
            entram na representação por-janela e, com ``pool="attention"``, informam os
            pesos do pooling (janelas hesitantes ganham peso). Resolve o "esmagamento".
    """
    import torch
    from torch import nn

    fuse_tab = bool(use_tabular) and int(dim_tab) > 0
    token_tab = fuse_tab and tab_fusion == "token"  # tab entra por-janela (pré-pooling)
    late_tab = fuse_tab and tab_fusion == "late"  # tab concatenado pós-pooling (legado)

    class _CrossAttentionFusionModule(nn.Module):
        """Fusão cross-attention sobre sequência de janelas ``(B, T, D)`` com masking."""

        def __init__(self) -> None:
            super().__init__()
            self.fuse_tab = fuse_tab
            self.token_tab = token_tab
            self.late_tab = late_tab
            self.pool = pool
            self.proj_a = nn.Linear(dim_a, common_dim)  # áudio -> common
            self.proj_b = nn.Linear(dim_b, common_dim)  # texto -> common
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(common_dim)
            if self.fuse_tab:
                # Projeção do tab p/ common_dim (usada por 'late' e 'token').
                # BatchNorm no INPUT padroniza cada feature (escalas heterogêneas: f0~500,
                # contagens~2, probs~0.5) — sem isso o proj_tab é dominado pelas de grande
                # magnitude. É per-feature (ao contrário do LayerNorm, que é por-amostra).
                self.tab_in_norm = nn.BatchNorm1d(dim_tab)
                self.proj_tab = nn.Linear(dim_tab, common_dim)
                self.tab_norm = nn.LayerNorm(common_dim)
            if self.token_tab:
                # Funde [token áudio-texto ‖ token tab] -> common_dim por janela, e
                # re-normaliza (LN) antes do pooling.
                self.token_fuse = nn.Linear(common_dim * 2, common_dim)
                self.token_norm = nn.LayerNorm(common_dim)
            if self.pool == "attention":
                # Pooling atenção-MIL gated (Ilse et al. 2018): score por janela via
                # gate tanh×sigmoid, softmax mascarado sobre T, soma ponderada.
                self.attn_V = nn.Linear(common_dim, common_dim)
                self.attn_U = nn.Linear(common_dim, common_dim)
                self.attn_w = nn.Linear(common_dim, 1)
            head_in = common_dim * 2 if self.late_tab else common_dim
            self.classifier = nn.Sequential(
                nn.Linear(head_in, common_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(common_dim, 1),  # 1 logit por vídeo
            )

        def forward(
            self,
            feat_a: torch.Tensor,  # (B, T, dim_a) — áudio por janela
            feat_b: torch.Tensor,  # (B, T, dim_b) — texto por janela
            key_padding_mask: torch.Tensor | None = None,  # (B, T) True = padding
            feat_tab: torch.Tensor | None = None,  # (B, T, dim_tab) — tabular por janela
        ) -> torch.Tensor:
            """Devolve ``(B, 1)`` logits a nível de vídeo.

            ``key_padding_mask[b, t] = True`` indica janela de padding (variável T).
            ``feat_tab`` só é usado quando ``use_tabular`` (senão é ignorado).
            """
            q = self.proj_a(feat_a)  # (B, T, C)  query = áudio
            kv = self.proj_b(feat_b)  # (B, T, C)  key/value = texto
            attn_out, _ = self.cross_attn(
                q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
            )
            fused = self.norm(q + attn_out)  # residual + LN -> (B, T, C)

            if self.fuse_tab and feat_tab is None:
                raise ValueError(
                    "use_tabular=True mas 'tab_seq' não veio no batch — o modelo espera "
                    "as features de suporte por janela."
                )
            # FUSÃO TABULAR POR-TOKEN (pré-pooling): o tab entra na representação de cada
            # janela ANTES da agregação, então participa dos pesos do pooling atenção-MIL.
            if self.token_tab:
                tab_tok = self._tab_tokens(feat_tab, key_padding_mask)  # (B, T, C)
                fused = self.token_norm(self.token_fuse(torch.cat([fused, tab_tok], dim=-1)))

            pooled = self._pool(fused, key_padding_mask)  # (B, C) — agregação temporal

            # FUSÃO TABULAR LATE (pós-pooling, legado): ramo paralelo concatenado.
            if self.late_tab:
                tab_pooled = self._masked_mean(feat_tab, key_padding_mask)  # (B, dim_tab)
                tab_pooled = self.tab_in_norm(tab_pooled)  # padroniza cada feature (escala)
                tab_repr = self.tab_norm(torch.relu(self.proj_tab(tab_pooled)))  # (B, C)
                pooled = torch.cat([pooled, tab_repr], dim=1)  # (B, 2C)
            return self.classifier(pooled)  # (B, 1)

        # ---- agregação temporal (janela → vídeo) --------------------------
        def _pool(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            """Despacha o pooling temporal conforme ``self.pool``."""
            if self.pool == "attention":
                return self._masked_attention_pool(x, mask)
            if self.pool == "max":
                return self._masked_max(x, mask)
            return self._masked_mean(x, mask)

        def _masked_attention_pool(
            self, x: torch.Tensor, mask: torch.Tensor | None
        ) -> torch.Tensor:
            """Pooling atenção-MIL gated (Ilse et al. 2018), mascarado sobre T."""
            gate = torch.tanh(self.attn_V(x)) * torch.sigmoid(self.attn_U(x))  # (B, T, C)
            scores = self.attn_w(gate).squeeze(-1)  # (B, T)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))  # janelas de padding: peso 0
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
            return (weights * x).sum(dim=1)  # (B, C) — soma ponderada

        @staticmethod
        def _masked_max(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            """Máximo temporal mascarado sobre T (janela mais forte manda)."""
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
            return x.max(dim=1).values

        @staticmethod
        def _masked_mean(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
            """Pooling temporal mascarado sobre T (média ignorando padding)."""
            if key_padding_mask is None:
                return x.mean(dim=1)
            valid = (~key_padding_mask).unsqueeze(-1).float()  # (B, T, 1)
            summed = (x * valid).sum(dim=1)  # (B, C)
            count = valid.sum(dim=1).clamp_min(1.0)  # (B, 1)
            return summed / count

        # ---- projeção tabular por-token (padroniza sem contaminar padding) ----
        def _tab_tokens(self, feat_tab: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            """Projeta o tab de cada janela p/ common_dim: BN per-feature (só tokens válidos)
            → Linear → ReLU → LN. Retorna ``(B, T, C)``."""
            b, t, d = feat_tab.shape
            flat = feat_tab.reshape(b * t, d)  # (B*T, dim_tab)
            normed = flat
            if mask is not None:
                valid = (~mask).reshape(b * t)  # (B*T,) True = janela real
                if valid.any():
                    normed = flat.clone()
                    # BatchNorm só nas janelas reais → estatísticas sem contaminação de padding.
                    normed[valid] = self.tab_in_norm(flat[valid])
            else:
                normed = self.tab_in_norm(flat)
            tok = self.tab_norm(torch.relu(self.proj_tab(normed)))  # (B*T, C)
            return tok.reshape(b, t, -1)

    return _CrossAttentionFusionModule()


# =============================================================================
# Fachada registrável (sklearn-like .from_config) — sem herdar nn.Module
# =============================================================================
class CrossAttentionFusion:
    """Fachada da cross-attention para o registry (family="lightning").

    Guarda os hiperparâmetros e constrói o ``nn.Module`` *lazy* via ``build_module``.
    O ``LightningTrainer`` chama ``build_lightning_module`` para obter o
    ``LightningModule`` treinável (``LitCrossAttention``).
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.dim_a = int(cfg.get("dim_a", 768))  # áudio (wav2vec2/hubert)
        self.dim_b = int(cfg.get("dim_b", 768))  # texto (roberta-emotion)
        self.dim_tab = int(cfg.get("dim_tab") or 0)  # tabular/hesitação (inferido do cache)
        self.use_tabular = bool(cfg.get("use_tabular", False))  # funde tab_seq (opcional)
        self.pool = str(cfg.get("pool", "mean"))  # agregação temporal: mean|attention|max
        self.tab_fusion = str(cfg.get("tab_fusion", "late"))  # tab: late (concat) | token (pré-pool)
        self.common_dim = int(cfg.get("common_dim", 512))
        self.num_heads = int(cfg.get("num_heads", 4))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-2))

    @classmethod
    def from_config(cls, config: Any) -> CrossAttentionFusion:
        """Lê ``configs/model/cross_attention.yaml`` (README §7, grupo ``model``)."""
        return cls(cfg=config)

    def build_module(self):
        """Instancia o ``nn.Module`` de fusão (importa torch *lazy*)."""
        return _build_fusion_module(
            dim_a=self.dim_a,
            dim_b=self.dim_b,
            common_dim=self.common_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            dim_tab=self.dim_tab,
            use_tabular=self.use_tabular,
            pool=self.pool,
            tab_fusion=self.tab_fusion,
        )

    def build_lightning_module(self):
        """Monta o ``LitCrossAttention`` pronto para ``L.Trainer.fit`` (lazy)."""
        return _build_lit_module(
            fusion=self.build_module(), lr=self.lr, weight_decay=self.weight_decay
        )

    def load_from_checkpoint(self, ckpt_path, map_location=None):
        """Reconstrói o ``nn.Module`` de fusão e carrega os pesos de um ``.ckpt`` Lightning.

        Helper de inferência (FASE 5 ``_predict_neural``). Como o ``LitCrossAttention`` é
        uma classe **closure** (definida dentro de ``_build_lit_module``, sem
        ``save_hyperparameters``) e **não** é exportada, ``LitCrossAttention.load_from_checkpoint``
        não é alcançável; aqui reconstruímos o ``nn.Module`` a partir do ``cfg`` desta fachada
        e carregamos o ``state_dict`` do checkpoint. Mantém ``torch``/``lightning`` *lazy*.

        O ``LitCrossAttention`` guarda o módulo de fusão em ``self.model`` (ver
        ``_build_lit_module``), então as chaves do ``state_dict`` têm prefixo ``model.``.

        Args:
            ckpt_path: caminho do arquivo ``.ckpt`` salvo pelo ``ModelCheckpoint``.
            map_location: device de destino (``torch.device`` ou str) p/ ``torch.load``.

        Returns:
            ``nn.Module`` de fusão em modo ``eval()`` — callable como
            ``module(feat_a, feat_b, key_padding_mask=...)`` → ``(B, 1)`` logits.
        """
        import torch

        ckpt = torch.load(str(ckpt_path), map_location=map_location)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        module = self.build_module()
        # O LightningModule expõe a fusão em ``self.model`` → remove o prefixo "model.".
        prefix = "model."
        fusion_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        # Fallback: se não houver prefixo (ckpt já só com a fusão), usa o state_dict inteiro.
        module.load_state_dict(fusion_state or state_dict)
        module.eval()
        return module


# =============================================================================
# LightningModule (construído lazy) — reusa evaluation/metrics.py do matheus
# =============================================================================
def _build_lit_module(fusion, lr: float, weight_decay: float):
    """Constrói o ``LitCrossAttention`` (importa lightning/torch/torchmetrics *lazy*).

    Reusa o ``ClassificationMetrics`` (torchmetrics: BinaryAveragePrecision +
    MulticlassF1Score macro) portado de ``src/evaluation/metrics.py`` do ``matheus``.
    """
    import lightning as L
    import torch
    from torch import nn, optim
    from torchmetrics.classification import BinaryAveragePrecision, MulticlassF1Score

    class _ClassificationMetrics(nn.Module):
        """Métricas torchmetrics do caminho Lightning (reuso do matheus)."""

        def __init__(self) -> None:
            super().__init__()
            self.val_ap = BinaryAveragePrecision()
            self.val_f1_macro = MulticlassF1Score(num_classes=2, average="macro")
            self.test_ap = BinaryAveragePrecision()
            self.test_f1_macro = MulticlassF1Score(num_classes=2, average="macro")

    class LitCrossAttention(L.LightningModule):
        """LightningModule da cross-attention temporal (1 logit por vídeo, BCE)."""

        def __init__(self) -> None:
            super().__init__()
            self.model = fusion
            self.metrics = _ClassificationMetrics()
            self._lr = lr
            self._weight_decay = weight_decay

        # ---- passos -------------------------------------------------------
        def _shared_step(self, batch):
            """Batch do ``VideoSequenceDataset``: tensores empilhados (B, T, D) + máscara."""
            feat_a = batch["audio_seq"]  # (B, T, dim_a)
            feat_b = batch["text_seq"]  # (B, T, dim_b)
            feat_tab = batch.get("tab_seq")  # (B, T, dim_tab) — usado só se use_tabular
            mask = batch["key_padding_mask"]  # (B, T) True = padding
            label = batch["label"].float()  # (B, 1) já vem do collate_sequences
            logit = self.model(feat_a, feat_b, key_padding_mask=mask, feat_tab=feat_tab)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, label)
            proba = torch.sigmoid(logit)  # (B, 1)
            return loss, proba, label.int()

        def training_step(self, batch, batch_idx):
            loss, _, _ = self._shared_step(batch)
            self.log("train_loss", loss, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            loss, proba, label = self._shared_step(batch)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            self.metrics.val_ap(proba, label)
            self.log("val_ap", self.metrics.val_ap, on_epoch=True, prog_bar=True)
            proba_2d = torch.cat([1 - proba, proba], dim=1)  # (B, 2)
            self.metrics.val_f1_macro(proba_2d, label.squeeze(1))
            self.log("val_f1_macro", self.metrics.val_f1_macro, on_epoch=True, prog_bar=True)

        def test_step(self, batch, batch_idx):
            loss, proba, label = self._shared_step(batch)
            self.log("test_loss", loss, on_epoch=True)
            self.metrics.test_ap(proba, label)
            self.log("test_ap", self.metrics.test_ap, on_epoch=True)
            proba_2d = torch.cat([1 - proba, proba], dim=1)
            self.metrics.test_f1_macro(proba_2d, label.squeeze(1))
            self.log("test_f1_macro", self.metrics.test_f1_macro, on_epoch=True)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            """Devolve ``(video_ids, proba)`` para agregação/calibração no trainer."""
            _, proba, _ = self._shared_step(batch)
            return {"video_ids": batch["video_id"], "proba": proba.squeeze(1)}

        # ---- otimização ---------------------------------------------------
        def configure_optimizers(self):
            optimizer = optim.AdamW(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }

    return LitCrossAttention()
