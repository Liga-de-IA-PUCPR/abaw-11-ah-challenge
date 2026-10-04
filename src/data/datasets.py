"""Datasets do BAH a partir do cache Parquet de janelas (README §6.2).

WindowMatrixView      — matriz achatada por janela p/ sklearn (random_forest, CPU, sem torch).
VideoSequenceDataset  — sequência de janelas por vídeo (T>1) p/ os modelos Lightning
                        (cross-attention, GNNs heterogêneos, GNN facial).
collate_sequences     — padding até T_max + key_padding_mask p/ T variável.

Colunas esperadas no Parquet (1 linha por janela):
    id, window_idx, t0, t1, participant_id, question_type,
    audio_emb (list<f32>), text_emb (list<f32>), tabular (list<f32>),
    label (i8, -1 se desconhecido), video_label (i8, -1 se test)
Coluna OPCIONAL (``mode=featurize_face``):
    face_landmarks (list<f32>, 478×3 achatado) → ``face_seq`` (T, 478, 3) no batch
Colunas EXTRAS (``mode=featurize_columns``, pedidas pelo modelo via ``data_spec``):
    qualquer coluna vetorial → ``features[<coluna>]`` (T, d) no batch; ``transcript`` →
    ``input_ids``/``attention_mask`` (transcrição tokenizada, ramo de texto fine-tunado)
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
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


# Colunas que toda visão lê (as extras entram sob demanda — o Parquet pode ter dezenas).
_BASE_COLUMNS = [
    "id", "window_idx", "participant_id", "audio_emb", "text_emb", "tabular", "label",
    "video_label",
]  # fmt: skip


def _read_window_parquet(
    parquet_path: str | Path,
    split_video_ids: set[str] | None = None,
    extra_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Lê do Parquet de janelas (Polars, scan preguiçoso) só as colunas necessárias.

    Args:
        parquet_path: caminho do ``text_audio_windows.parquet`` (FASE 3).
        split_video_ids: se dado, mantém apenas linhas cujo ``id`` está no conjunto
            (aplica o split participant-wise resolvido na FASE 4).
        extra_columns: colunas além das canônicas (ignora as que não existirem — quem as
            exige valida antes).
    """
    schema = pl.read_parquet_schema(parquet_path)
    cols = _BASE_COLUMNS + [c for c in extra_columns if c in schema and c not in _BASE_COLUMNS]
    lf = pl.scan_parquet(parquet_path).select(cols)
    if split_video_ids is not None:
        lf = lf.filter(pl.col("id").is_in(list(split_video_ids)))
    df = lf.collect()
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

        self.parquet_path: Path = Path(parquet_path)  # fonte (ensemble: dado por membro)
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

    def subset(self, video_ids: set[str]) -> WindowMatrixView:
        """Visão restrita às janelas dos vídeos ``video_ids`` (dobras do OOF, sem reler)."""
        rows = np.isin(self.video_ids, list(video_ids))
        sub = copy.copy(self)
        sub.X, sub.y, sub.groups = self.X[rows], self.y[rows], self.groups[rows]
        sub.video_ids = self.video_ids[rows]
        sub.video_labels = {v: lab for v, lab in self.video_labels.items() if v in video_ids}
        return sub


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
          "length": int, "label": int (0/1; -1 no test), "video_id": str,
          "face_seq": (T, 478, 3)   # só se o Parquet tiver a coluna face_landmarks
        }
    """

    def __init__(
        self,
        parquet_path: str | Path,
        split_video_ids: set[str] | None = None,
        columns: Sequence[str] | None = None,
        transcript: dict[str, Any] | None = None,
    ):
        """Args:
        parquet_path / split_video_ids: cache de janelas e vídeos do split.
        columns: colunas por janela extras (ex.: ``asr_timing``, ``audio_emb_<embedder>``)
            → ``features[<coluna>]`` no batch. As canônicas (``audio_emb``/``text_emb``/
            ``tabular``) também podem ser pedidas por nome.
        transcript: ``{"model_name", "max_length"}`` → tokeniza a coluna ``transcript``
            (texto do vídeo) com o tokenizer do modelo → ``input_ids``/``attention_mask``.
        """
        import torch  # noqa: F401 — lazy: falha cedo se o grupo `neural` não estiver instalado

        self.parquet_path: Path = Path(parquet_path)  # fonte (ensemble: dado por membro)
        self.columns: list[str] = list(columns or [])
        needed = self.columns + (["transcript"] if transcript else [])
        df = _read_window_parquet(parquet_path, split_video_ids, [*needed, "face_landmarks"])
        df = df.sort(["id", "window_idx"])
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(
                f"Colunas {missing} ausentes em {self.parquet_path.name} — grave-as com "
                f"'mode=featurize_columns columns=[...]' (ver src/features/columns.py)."
            )
        # Vídeo (opcional): landmarks do Face Mesh gravados por mode=featurize_face.
        self._has_face = "face_landmarks" in df.columns
        face_shape = (0, 0)
        if self._has_face:
            from src.features.face_mesh import LANDMARK_DIM, NUM_FACE_LANDMARKS

            face_shape = (NUM_FACE_LANDMARKS, LANDMARK_DIM)
        texts = self._group_videos(df, face_shape, with_text=bool(transcript))

        self.lengths: list[int] = [a.shape[0] for a in self._audio]
        # Dimensões dos embeddings no cache (librosa 320 / wav2vec2 768; texto 768).
        # O LightningTrainer infere as dims do modelo daqui (sem hardcode na config).
        self.dim_audio: int = int(self._audio[0].shape[1]) if self._audio else 0
        self.dim_text: int = int(self._text[0].shape[1]) if self._text else 0
        self.dim_tab: int = int(self._tab[0].shape[1]) if self._tab else 0
        self.has_face: bool = self._has_face
        self.dim_face: tuple[int, int] = face_shape
        # {coluna: d} de todas as colunas por janela carregadas (canônicas + extras).
        self.dims: dict[str, int] = {
            "audio_emb": self.dim_audio,
            "text_emb": self.dim_text,
            "tabular": self.dim_tab,
            **{c: int(v[0].shape[1]) for c, v in self._extra.items() if v},
        }
        self._tokens: list[list[int]] = []
        self.pad_token_id = 0
        if transcript:
            self._tokens, self.pad_token_id = _tokenize(texts, **transcript)
        # Acessor público alinhado com WindowMatrixView (FASE_4 depende deste contrato):
        # {video_id: global_ah} (rótulo a nível de vídeo; -1 = test).
        self.video_labels: dict[str, int] = dict(zip(self.video_ids, self._labels, strict=False))
        log.info(
            f"VideoSequenceDataset: {len(self.video_ids)} vídeos, "
            f"T∈[{min(self.lengths)}, {max(self.lengths)}], "
            f"T_max={max(self.lengths)}"
            + (f", colunas extras={self.columns}" if self.columns else "")
            + (" + transcrição tokenizada" if transcript else "")
        )

    def _group_videos(
        self, df: pl.DataFrame, face_shape: tuple[int, int], with_text: bool
    ) -> list[str]:
        """Empilha as janelas de cada vídeo (ordem temporal) → listas por vídeo; devolve as
        transcrições (``with_text``) p/ tokenizar."""
        self.video_ids: list[str] = []
        self._audio: list[np.ndarray] = []
        self._text: list[np.ndarray] = []
        self._tab: list[np.ndarray] = []
        self._face: list[np.ndarray] = []
        self._labels: list[int] = []
        self._window_labels: list[np.ndarray] = []
        canonical = {"audio_emb": self._audio, "text_emb": self._text, "tabular": self._tab}
        self._extra: dict[str, list[np.ndarray]] = {c: canonical.get(c, []) for c in self.columns}
        texts: list[str] = []
        for vid, g in df.group_by("id", maintain_order=True):
            vid = vid[0] if isinstance(vid, tuple) else vid
            self.video_ids.append(str(vid))
            self._audio.append(np.vstack(g["audio_emb"].to_numpy()).astype(np.float32))
            self._text.append(np.vstack(g["text_emb"].to_numpy()).astype(np.float32))
            self._tab.append(np.vstack(g["tabular"].to_numpy()).astype(np.float32))
            for c in self.columns:
                if c not in canonical:
                    self._extra[c].append(np.vstack(g[c].to_numpy()).astype(np.float32))
            if self._has_face:
                face_flat = np.vstack(g["face_landmarks"].to_numpy()).astype(np.float32)
                self._face.append(face_flat.reshape(-1, *face_shape))
            self._labels.append(int(g["video_label"][0]))
            self._window_labels.append(g["label"].to_numpy().astype(np.int64))
            if with_text:
                texts.append(str(g["transcript"][0] or ""))
        return texts

    def __len__(self) -> int:
        return len(self.video_ids)

    def subset(self, video_ids: set[str]) -> VideoSequenceDataset:
        """Visão dos vídeos ``video_ids`` compartilhando os arrays (dobras do OOF: o Parquet
        é lido uma vez só)."""
        keep = [i for i, v in enumerate(self.video_ids) if v in video_ids]

        def pick(xs: list) -> list:
            return [xs[i] for i in keep] if xs else xs

        sub = copy.copy(self)
        for name in ("video_ids", "_audio", "_text", "_tab", "_face", "_labels",
                     "_window_labels", "lengths", "_tokens"):  # fmt: skip
            setattr(sub, name, pick(getattr(self, name)))
        sub._extra = {c: pick(v) for c, v in self._extra.items()}
        sub.video_labels = dict(zip(sub.video_ids, sub._labels, strict=False))
        return sub

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Import local (não um atributo): o dataset precisa ser picklável p/ DataLoader com
        # num_workers>0 no macOS/Windows (start method "spawn" serializa o dataset).
        import torch

        item = {
            "audio_seq": torch.tensor(self._audio[idx], dtype=torch.float32),  # (T, d_a)
            "text_seq": torch.tensor(self._text[idx], dtype=torch.float32),  # (T, d_b)
            "tab_seq": torch.tensor(self._tab[idx], dtype=torch.float32),  # (T, d_tab)
            "length": self.lengths[idx],
            "label": self._labels[idx],
            "video_id": self.video_ids[idx],
        }
        if self._has_face:
            item["face_seq"] = torch.tensor(self._face[idx], dtype=torch.float32)  # (T, 478, 3)
        item["window_label"] = torch.tensor(self._window_labels[idx])  # (T,) — -1 = sem rótulo
        if self.columns:
            item["features"] = {
                c: torch.tensor(v[idx], dtype=torch.float32) for c, v in self._extra.items()
            }
        if self._tokens:
            item["input_ids"] = torch.tensor(self._tokens[idx], dtype=torch.long)
            item["pad_token_id"] = self.pad_token_id
        return item


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
          "video_id": list[str],
          "face_seq": (B, T_max, 478, 3),  # só se os itens tiverem face_seq
          "window_label": (B, T_max) long  # -1 = sem rótulo de janela / padding
          "features": {coluna: (B, T_max, d)},                  # colunas extras pedidas
          "input_ids"/"attention_mask": (B, L_max)              # transcrição tokenizada
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

    out: dict[str, Any] = {
        "audio_seq": audio,
        "text_seq": text,
        "tab_seq": tab,
        "lengths": lengths,
        "key_padding_mask": key_padding_mask,
        "label": labels,
        "video_id": [b["video_id"] for b in batch],
    }
    if "face_seq" in batch[0]:
        out["face_seq"] = pad_sequence([b["face_seq"] for b in batch], batch_first=True)
    if "window_label" in batch[0]:
        out["window_label"] = pad_sequence(
            [b["window_label"] for b in batch], batch_first=True, padding_value=-1
        )
    if "features" in batch[0]:
        out["features"] = {
            c: pad_sequence([b["features"][c] for b in batch], batch_first=True)
            for c in batch[0]["features"]
        }
    if "input_ids" in batch[0]:
        ids = [b["input_ids"] for b in batch]
        out["input_ids"] = pad_sequence(
            ids, batch_first=True, padding_value=int(batch[0]["pad_token_id"])
        )
        out["attention_mask"] = pad_sequence(
            [torch.ones_like(t) for t in ids], batch_first=True, padding_value=0
        )
    return out


def _tokenize(texts: list[str], model_name: str, max_length: int = 256) -> tuple[list, int]:
    """Tokeniza as transcrições (sem padding — o collate faz) → ``(ids por vídeo, pad_id)``."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok(texts, truncation=True, max_length=int(max_length))
    return list(enc["input_ids"]), int(tok.pad_token_id or 0)


def as_loader(cfg: DictConfig, data, family: str, split: str):
    """Adapta ``data`` ao que o trainer da ``family`` espera (FASE 6).

    O caminho ``sklearn`` consome ``WindowMatrixView`` direto (CPU, sem torch) →
    devolve ``data`` inalterado. O caminho ``lightning`` precisa de um
    ``DataLoader`` com ``collate_fn=collate_sequences`` (padding até T_max +
    ``key_padding_mask``); sem ele o ``LightningTrainer`` recebe um ``Dataset``
    cru (``_labels_from_loader`` quebra em ``loader.dataset`` e o
    ``_shared_step`` quebra na chave ausente ``key_padding_mask``).

    O import do ``torch`` é **lazy** para preservar o
    caminho ``sklearn`` 100% sem torch (README §7).
    """
    if family != "lightning":
        return data

    from torch.utils.data import DataLoader

    # Hard mining (opcional): data.hard_examples=<json de mode=hard_mining> troca o
    # shuffle uniforme do treino por um WeightedRandomSampler (mutuamente exclusivos).
    sampler = None
    hard_path = cfg.data.get("hard_examples")
    if split == "train" and hard_path:
        sampler = _build_weighted_sampler(hard_path, data)

    return DataLoader(
        data,
        batch_size=cfg.data.batch_size,
        shuffle=(split == "train") and sampler is None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_sequences,
    )


def _build_weighted_sampler(hard_path: str, dataset):
    """``WeightedRandomSampler`` alinhado à ordem de ``dataset.video_ids`` (hard mining)."""
    import json
    from pathlib import Path

    p = Path(hard_path)
    if not p.exists():
        log.warning(f"hard_examples ausente ({p}); amostragem uniforme.")
        return None

    from torch.utils.data import WeightedRandomSampler

    payload = json.loads(p.read_text(encoding="utf-8"))
    weights_map = payload.get("weights", payload)
    video_ids = getattr(dataset, "video_ids", None)
    if not video_ids:
        log.warning("Dataset sem video_ids; amostragem uniforme.")
        return None
    weights = [float(weights_map.get(str(vid), 1.0)) for vid in video_ids]
    log.info(
        f"WeightedRandomSampler: {len(weights)} amostras, "
        f"peso∈[{min(weights):.2f}, {max(weights):.2f}] de {p}"
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ==============================================================================
# Carregadores por split/família — consumidos pela FASE 6 (main.py)
# ==============================================================================


def _split_video_ids(df: pl.DataFrame, split: str | Sequence[str]) -> set[str]:
    """Resolve os ``id`` (vídeos) de um split (ou UNIÃO de splits) do Parquet de janelas.

    O split é **participant-wise**; o Parquet de features (FASE 3) carrega o split
    de cada janela na coluna ``split`` (derivada do ``VideoRecord`` na indexação).
    Aceita uma string única (``"train"``) OU uma sequência (``["train", "val"]``) —
    a união é usada para compor o conjunto de treino (splits são disjuntos por
    participante, então combinar não vaza).
    """
    splits = [split] if isinstance(split, str) else list(split)
    return set(df.filter(pl.col("split").is_in(splits))["id"].unique().to_list())


def video_table(parquet_path: str | Path, splits: Sequence[str]):
    """Uma linha por vídeo dos ``splits`` (``video_id``, ``participant_id``, ``label``),
    ordenada por ``video_id`` — a tabela que define as dobras/holdouts por participante."""
    import pandas as pd

    df = (
        pl.scan_parquet(parquet_path)
        .select("id", "participant_id", "video_label", "split")
        .filter(pl.col("split").is_in(list(splits)))
        .unique("id", keep="first")
        .sort("id")
        .collect()
    )
    return pd.DataFrame(
        {
            "video_id": df["id"].to_list(),
            "participant_id": df["participant_id"].to_list(),
            "label": df["video_label"].cast(pl.Int64).to_list(),
        }
    )


def _view_for_family(
    parquet_path: str | Path, family: str, split_ids: set[str], spec: dict | None = None
):
    """Devolve a visão correta dos dados por família (README §6.4 / FASE 4).

    - ``sklearn``   → :class:`WindowMatrixView` (matriz achatada por janela, CPU).
    - ``lightning`` → :class:`VideoSequenceDataset` (sequência ``(T, d)`` por vídeo), com as
      entradas extras que o modelo declara em ``spec`` (``columns``/``transcript``).
    """
    if family == "sklearn":
        return WindowMatrixView(parquet_path, split_video_ids=split_ids)
    if family == "lightning":
        return VideoSequenceDataset(parquet_path, split_video_ids=split_ids, **(spec or {}))
    raise ValueError(f"família desconhecida: {family!r} (use 'sklearn' | 'lightning').")


def model_data_spec(cfg: DictConfig) -> dict:
    """Entradas extras que o modelo de ``cfg.model`` pede ao dataset (``{}`` p/ os demais)."""
    from src.models.registry import data_spec

    return data_spec(cfg.model)


def load_split(
    cfg: DictConfig,
    split: str | Sequence[str],
    *,
    family: str,
    parquet_path: str | Path | None = None,
):
    """Carrega UM split (ou união de splits) do cache Parquet (FASE 3) na visão da ``family``.

    Args:
        cfg: config Hydra composto (usa ``cfg.data.paths.parquet_path`` por padrão).
        split: "train" | "val" | "test", ou uma sequência (ex.: ``["train", "val"]``).
        family: "sklearn" (WindowMatrixView) | "lightning" (VideoSequenceDataset).
        parquet_path: sobrepõe o Parquet lido (ex.: calibrar num Parquet diferente do
            de predição). ``None`` = usa ``cfg.data.paths.parquet_path``.

    Returns:
        :class:`WindowMatrixView` ou :class:`VideoSequenceDataset` do(s) split(s) pedido(s).
    """
    pq = Path(parquet_path) if parquet_path is not None else Path(cfg.data.paths.parquet_path)
    split_ids = _split_video_ids(pl.read_parquet(pq, columns=["id", "split"]), split)
    log.info(f"load_split(split={split}, family={family}): {len(split_ids)} vídeos [{pq.name}]")
    return load_videos(cfg, split_ids, family=family, parquet_path=pq)


def load_videos(
    cfg: DictConfig,
    video_ids: set[str],
    *,
    family: str,
    parquet_path: str | Path | None = None,
):
    """Visão da ``family`` sobre um conjunto EXPLÍCITO de vídeos (folds do OOF, holdout)."""
    pq = Path(parquet_path) if parquet_path is not None else Path(cfg.data.paths.parquet_path)
    spec = model_data_spec(cfg) if family == "lightning" else None
    return _view_for_family(pq, family, set(video_ids), spec)


def load_train_val(cfg: DictConfig, *, family: str):
    """Carrega os splits de treino e de calibração na visão da ``family``.

    Atalho usado por ``mode=train`` (FASE 6): devolve ``(train_data, calib_data)`` já
    na visão certa (matriz p/ sklearn, sequência p/ Lightning). Configurável:
    - ``data.train_splits`` (default ``[train]``): splits UNIDOS para treinar (ex.:
      ``[train, val]`` p/ usar toda a base rotulada quando a avaliação real é externa).
    - ``data.calib_split`` (default ``val``): split usado para calibrar o limiar (e, no
      caminho neural, para monitorar early-stop/checkpoint). Ex.: ``test`` (525, grande
      e limpo) — calibra o limiar num conjunto robusto sem vazamento (splits disjuntos).

    Os defaults reproduzem EXATAMENTE o comportamento anterior (train / val).

    Args:
        cfg: config Hydra composto.
        family: "sklearn" | "lightning".

    Returns:
        Tupla ``(train_data, calib_data)``.
    """
    data = cfg.data
    train_splits = data.get("train_splits", ["train"])
    train_splits = [train_splits] if isinstance(train_splits, str) else list(train_splits)
    calib_split = data.get("calib_split", "val")
    log.info(f"Composição de splits: treino={train_splits} · calibração/monitor='{calib_split}'")
    return (
        load_split(cfg, train_splits, family=family),
        load_split(cfg, calib_split, family=family),
    )
