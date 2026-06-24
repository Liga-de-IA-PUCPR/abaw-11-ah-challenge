"""Datasets do BAH a partir do cache Parquet de janelas (README §6.2).

WindowMatrixView      — matriz achatada por janela p/ sklearn (random_forest, CPU, sem torch).
VideoSequenceDataset  — sequência de janelas por vídeo (T>1) p/ a cross-attention (Lightning).
collate_sequences     — padding até T_max + key_padding_mask p/ T variável.

Colunas esperadas no Parquet (1 linha por janela):
    id, window_idx, t0, t1, participant_id, question_type,
    audio_emb (list<f32>), text_emb (list<f32>), tabular (list<f32>),
    label (i8, -1 se desconhecido), video_label (i8, -1 se test)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from omegaconf import DictConfig

from src.logger import get_logger

if TYPE_CHECKING:  # evita importar torch no caminho sklearn/CPU
    from torch.utils.data import Dataset
else:  # base dummy p/ não forçar dependência neural no import do módulo
    Dataset = object

log = get_logger("data.datasets")


# ==============================================================================
# Leitura do Parquet (compartilhada)
# ==============================================================================


def _read_window_parquet(
    parquet_path: str | Path,
    split_video_ids: set[str] | None = None,
) -> pl.DataFrame:
    """Lê o Parquet de janelas (Polars) e, opcionalmente, filtra por ``id`` (vídeo).

    Args:
        parquet_path: caminho do ``text_audio_windows.parquet`` (FASE 3).
        split_video_ids: se dado, mantém apenas linhas cujo ``id`` está no conjunto
            (aplica o split participant-wise resolvido na FASE 4).
    """
    df = pl.read_parquet(parquet_path)
    if split_video_ids is not None:
        df = df.filter(pl.col("id").is_in(list(split_video_ids)))
    log.info(f"Parquet: {df.height} janelas, {df['id'].n_unique()} vídeos")
    return df


# ==============================================================================
# (a) Matriz achatada por janela — random_forest (sklearn, CPU)
# ==============================================================================


class WindowMatrixView:
    """Visão matricial das janelas para o ``random_forest`` (família sklearn).

    100% NumPy/Polars — **não importa torch nem torchmetrics**. Concatena por janela
    ``X = [audio_emb ‖ text_emb ‖ tabular]`` e expõe os vetores que a FASE 4 precisa
    para treinar (rótulo de janela) e depois **agregar janela→vídeo** (``video_ids`` +
    ``video_labels``) com limiar calibrado.

    Attributes:
        X: ``(n_janelas, d_audio + d_text + d_tab)`` float32.
        y: ``(n_janelas,)`` int — rótulo da janela (-1 = desconhecido).
        groups: ``(n_janelas,)`` — ``participant_id`` (split participant-wise).
        video_ids: ``(n_janelas,)`` — ``id`` do vídeo de cada janela (p/ agregação).
        video_labels: ``{video_id: global_ah}`` (rótulo a nível de vídeo; -1 = test).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        split_video_ids: set[str] | None = None,
    ):
        df = _read_window_parquet(parquet_path, split_video_ids)
        # Ordena por (id, window_idx) p/ reprodutibilidade.
        df = df.sort(["id", "window_idx"])

        audio = np.vstack(df["audio_emb"].to_numpy()).astype(np.float32)
        text = np.vstack(df["text_emb"].to_numpy()).astype(np.float32)
        tab = np.vstack(df["tabular"].to_numpy()).astype(np.float32)

        self.X: np.ndarray = np.concatenate([audio, text, tab], axis=1)
        self.y: np.ndarray = df["label"].to_numpy().astype(np.int64)
        self.groups: np.ndarray = df["participant_id"].to_numpy()
        self.video_ids: np.ndarray = df["id"].to_numpy()

        # Um rótulo de vídeo por id (primeiro valor; constante dentro do grupo).
        vl = df.group_by("id").agg(pl.col("video_label").first())
        self.video_labels: dict[str, int] = dict(
            zip(vl["id"].to_list(), vl["video_label"].to_list(), strict=False)
        )
        self.dims: dict[str, int] = {
            "audio": audio.shape[1],
            "text": text.shape[1],
            "tabular": tab.shape[1],
        }
        log.info(
            f"WindowMatrixView: X={self.X.shape}, vídeos={len(self.video_labels)}, dims={self.dims}"
        )

    def __len__(self) -> int:
        return self.X.shape[0]


# ==============================================================================
# (b) Sequência por vídeo (T>1) — cross_attention (Lightning)
# ==============================================================================


class VideoSequenceDataset(Dataset):
    """Sequência de janelas por vídeo (T>1) — eleva o EmbeddingDataset (matheus, T=1).

    Cada item é UM vídeo: empilha suas janelas (ordenadas por ``window_idx``) em
    sequências ``(T, d_*)`` e devolve o rótulo a nível de vídeo (``video_label`` =
    ``global_ah``). O padding até ``T_max`` e o ``key_padding_mask`` são feitos no
    ``collate_sequences`` (T varia por vídeo).

    Importa torch **lazy** (só este caminho usa o grupo opcional ``neural``).

    Item (antes do collate):
        {
          "audio_seq": (T, d_a), "text_seq": (T, d_b), "tab_seq": (T, d_tab),
          "length": int, "label": int (0/1; -1 no test), "video_id": str
        }
    """

    def __init__(
        self,
        parquet_path: str | Path,
        split_video_ids: set[str] | None = None,
    ):
        import torch  # lazy: caminho neural

        self._torch = torch
        df = _read_window_parquet(parquet_path, split_video_ids)
        df = df.sort(["id", "window_idx"])

        # Agrupa janelas por vídeo preservando a ordem temporal.
        self.video_ids: list[str] = []
        self._audio: list[np.ndarray] = []
        self._text: list[np.ndarray] = []
        self._tab: list[np.ndarray] = []
        self._labels: list[int] = []

        for vid, g in df.group_by("id", maintain_order=True):
            vid = vid[0] if isinstance(vid, tuple) else vid
            self.video_ids.append(str(vid))
            self._audio.append(np.vstack(g["audio_emb"].to_numpy()).astype(np.float32))
            self._text.append(np.vstack(g["text_emb"].to_numpy()).astype(np.float32))
            self._tab.append(np.vstack(g["tabular"].to_numpy()).astype(np.float32))
            self._labels.append(int(g["video_label"][0]))

        self.lengths: list[int] = [a.shape[0] for a in self._audio]
        # Acessor público alinhado com WindowMatrixView (FASE_4 depende deste contrato):
        # {video_id: global_ah} (rótulo a nível de vídeo; -1 = test).
        self.video_labels: dict[str, int] = dict(zip(self.video_ids, self._labels, strict=False))
        log.info(
            f"VideoSequenceDataset: {len(self.video_ids)} vídeos, "
            f"T∈[{min(self.lengths)}, {max(self.lengths)}], "
            f"T_max={max(self.lengths)}"
        )

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch = self._torch
        return {
            "audio_seq": torch.tensor(self._audio[idx], dtype=torch.float32),  # (T, d_a)
            "text_seq": torch.tensor(self._text[idx], dtype=torch.float32),  # (T, d_b)
            "tab_seq": torch.tensor(self._tab[idx], dtype=torch.float32),  # (T, d_tab)
            "length": self.lengths[idx],
            "label": self._labels[idx],
            "video_id": self.video_ids[idx],
        }


def collate_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate p/ ``VideoSequenceDataset``: padding até T_max + ``key_padding_mask``.

    Empilha um lote de vídeos com ``T`` variável em tensores ``(B, T_max, d_*)``,
    zero-pad à direita. O ``key_padding_mask`` (``(B, T_max)``, ``True`` = posição
    de padding) é consumido pela ``MultiheadAttention`` da cross-attention (FASE 4),
    garantindo que janelas-fantasma não contaminem a atenção nem o pooling temporal.

    Returns:
        {
          "audio_seq": (B, T_max, d_a), "text_seq": (B, T_max, d_b),
          "tab_seq": (B, T_max, d_tab), "lengths": (B,),
          "key_padding_mask": (B, T_max) bool, "label": (B, 1) float,
          "video_id": list[str]
        }
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    audio = pad_sequence([b["audio_seq"] for b in batch], batch_first=True)
    text = pad_sequence([b["text_seq"] for b in batch], batch_first=True)
    tab = pad_sequence([b["tab_seq"] for b in batch], batch_first=True)

    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    t_max = int(audio.shape[1])
    # mask[b, t] = True quando t >= length[b] (posição de padding).
    ar = torch.arange(t_max).unsqueeze(0)  # (1, T_max)
    key_padding_mask = ar >= lengths.unsqueeze(1)  # (B, T_max)

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32).unsqueeze(
        1
    )  # (B, 1) p/ BCEWithLogits

    return {
        "audio_seq": audio,
        "text_seq": text,
        "tab_seq": tab,
        "lengths": lengths,
        "key_padding_mask": key_padding_mask,
        "label": labels,
        "video_id": [b["video_id"] for b in batch],
    }


# ==============================================================================
# Carregadores por split/família — consumidos pela FASE 6 (main.py)
# ==============================================================================


def _split_video_ids(df: pl.DataFrame, split: str) -> set[str]:
    """Resolve os ``id`` (vídeos) de um split a partir do Parquet de janelas.

    O split é **participant-wise**; o Parquet de features (FASE 3) carrega o split
    de cada janela na coluna ``split`` (derivada do ``VideoRecord`` na indexação).
    """
    return set(df.filter(pl.col("split") == split)["id"].unique().to_list())


def _view_for_family(parquet_path: str | Path, family: str, split_ids: set[str]):
    """Devolve a visão correta dos dados por família (README §6.4 / FASE 4).

    - ``sklearn``   → :class:`WindowMatrixView` (matriz achatada por janela, CPU).
    - ``lightning`` → :class:`VideoSequenceDataset` (sequência ``(T, d)`` por vídeo).
    """
    if family == "sklearn":
        return WindowMatrixView(parquet_path, split_video_ids=split_ids)
    if family == "lightning":
        return VideoSequenceDataset(parquet_path, split_video_ids=split_ids)
    raise ValueError(f"família desconhecida: {family!r} (use 'sklearn' | 'lightning').")


def load_split(cfg: DictConfig, split: str, *, family: str):
    """Carrega UM split do cache Parquet (FASE 3) na visão da ``family``.

    Args:
        cfg: config Hydra composto (usa ``cfg.data.paths.parquet_path``).
        split: "train" | "val" | "test".
        family: "sklearn" (WindowMatrixView) | "lightning" (VideoSequenceDataset).

    Returns:
        :class:`WindowMatrixView` ou :class:`VideoSequenceDataset` do split pedido.
    """
    parquet_path = Path(cfg.data.paths.parquet_path)
    df = pl.read_parquet(parquet_path)
    split_ids = _split_video_ids(df, split)
    log.info(f"load_split(split={split}, family={family}): {len(split_ids)} vídeos")
    return _view_for_family(parquet_path, family, split_ids)


def load_train_val(cfg: DictConfig, *, family: str):
    """Carrega os splits de treino e validação na visão da ``family``.

    Atalho usado por ``mode=train`` (FASE 6): devolve ``(train_data, val_data)`` já
    na visão certa (matriz p/ sklearn, sequência p/ Lightning).

    Args:
        cfg: config Hydra composto.
        family: "sklearn" | "lightning".

    Returns:
        Tupla ``(train_data, val_data)``.
    """
    return load_split(cfg, "train", family=family), load_split(cfg, "val", family=family)
