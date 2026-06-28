"""Embeddings de texto via HuggingFace (RoBERTa-emotion EN como default configurável).

⚠️ ALERTA DE IDIOMA (README §8): as transcrições do dataset BAH são em INGLÊS.
O default é ``cardiffnlp/twitter-roberta-base-emotion`` (RoBERTa EN, emoção). Um
encoder PORTUGUÊS (BERTimbau) degradaria os embeddings — só usar com texto traduzido.

O embedder é AGNÓSTICO AO MODELO: qualquer ``AutoModel`` do HuggingFace funciona.
A dimensão de saída é lida de ``model.config.hidden_size`` (768 base, 384 MiniLM, 1024 large).

Reusa o pooling do ``TextVectorizer`` da branch ``matheus`` (mean-pool mascarado pela
attention + L2-normalize), mas device-aware via ``resolve_device`` (CPU/MPS/CUDA).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.base.embedder import BaseEmbedder
from src.conf import resolve_device
from src.logger import get_logger

log = get_logger("features.text_embedder")

# Silencia o "LOAD REPORT" do transformers: carregar um ...ForSequenceClassification via
# AutoModel descarta a head de classificação (UNEXPECTED) e inicia um pooler novo (MISSING)
# — esperado e benigno, pois usamos mean-pooling sobre last_hidden_state, não a head/pooler.
hf_logging.set_verbosity_error()


class TextEmbedder(BaseEmbedder):
    """Extrai embeddings de texto de janelas usando um encoder HuggingFace.

    Espelha o contrato de :class:`BaseEmbedder` (README §6.3):
    ``extract(texts: list[str]) -> np.ndarray (n, hidden) float32``.

    Pooling suportado:
    - ``"mean"``: média dos tokens **mascarada pela attention** (ignora padding) +
      L2-normalize — idêntico ao ``TextVectorizer`` do matheus. Recomendado.
    - ``"cls"``: embedding do token ``[CLS]`` (primeira posição).

    Attributes:
        name: Identificador do embedder ("text").
        model_name: Nome/caminho do modelo HuggingFace.
        pooling: Estratégia de pooling ("mean" | "cls").
        max_length: Truncamento máximo de tokens por janela.
        batch_size: Tamanho do batch na inferência.
        normalize: Se aplica L2-normalize após o pooling (default True).
        device: ``torch.device`` resolvido (CPU/MPS/CUDA).
    """

    name: str = "text"

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-emotion",
        pooling: Literal["mean", "cls"] = "mean",
        max_length: int = 128,
        batch_size: int = 32,
        normalize: bool = True,
        device: str = "auto",
    ) -> None:
        """Inicializa o TextEmbedder.

        Args:
            model_name: Encoder HuggingFace. Default RoBERTa-emotion EN (ver §8).
            pooling: "mean" (mean-pool mascarado) ou "cls".
            max_length: Máx. de tokens (janelas de ~5 s raramente excedem 128).
            batch_size: Batch de inferência.
            normalize: L2-normalize após o pooling (como no matheus).
            device: "auto" (MPS▸CUDA▸CPU) | "cpu" | "mps" | "cuda".
        """
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = resolve_device(device)

        log.info(f"Carregando tokenizer/modelo de texto: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # AutoModel devolve o encoder base; a head de classificação é ignorada.
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # dim = hidden_size do encoder (768 base, 1024 large, 384 MiniLM).
        self._dim: int = int(self.model.config.hidden_size)

        # ⚠️ Aviso ativo de incompatibilidade de idioma com o dataset (EN).
        if "portuguese" in model_name.lower() or "bertimbau" in model_name.lower():
            log.warning(
                "TextEmbedder usando modelo PORTUGUÊS (%s), mas as transcrições "
                "do BAH são em INGLÊS. Considere um encoder EN/multilíngue (README §8).",
                model_name,
            )

        log.info(
            f"TextEmbedder pronto: dim={self._dim}, pooling={pooling}, "
            f"normalize={normalize}, device={self.device}"
        )

    # =========================================================================
    # Contrato BaseEmbedder
    # =========================================================================

    @property
    def dim(self) -> int:
        """Dimensão do embedding de texto (= hidden_size do encoder)."""
        return self._dim

    @torch.no_grad()
    def extract(self, inputs: list[str]) -> np.ndarray:
        """Extrai embeddings para uma lista de textos de janela.

        Args:
            inputs: Lista de strings (campo ``WindowSample.text``). Strings vazias
                são aceitas (janela sem transcrição sobreposta → vetor do encoder
                para o texto vazio).

        Returns:
            Array ``float32`` de shape ``(n, hidden)``.
        """
        if len(inputs) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)

        texts = [t if isinstance(t, str) and t.strip() else "" for t in inputs]

        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            hidden = self.model(**enc).last_hidden_state  # (b, seq, hidden)

            if self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:  # mean-pool mascarado (matheus)
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts

            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            out.append(pooled.detach().cpu().numpy().astype(np.float32))

        result = np.concatenate(out, axis=0)
        log.debug(f"TextEmbedder.extract: {result.shape[0]} janelas -> dim {result.shape[1]}")
        return result

    def feature_names(self) -> list[str]:
        """Nomes determinísticos: ['text_emb_0', ..., 'text_emb_{dim-1}']."""
        return [f"text_emb_{i}" for i in range(self._dim)]
