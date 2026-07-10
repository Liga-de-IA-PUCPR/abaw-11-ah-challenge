"""Structured configs (Hydra) + helper de device para o desafio BAH (áudio + texto).

Os dataclasses abaixo espelham os grupos YAML de ``configs/`` (README §4/§7) e são
registrados no :class:`hydra.core.config_store.ConfigStore`. Isso dá **validação e
autocompletar** sem trocar a interface: o **YAML continua sendo a fonte de verdade** e
o usuário compõe/sobrescreve via Hydra (defaults list, presets ``experiment/*``,
override CLI e multirun ``-m``).

Convenções:
- Campos opcionais usam ``MISSING`` (validado em runtime) ou ``None`` quando derivável.
- ``device`` é global; embedders e o ``LightningTrainer`` o resolvem com
  :func:`resolve_device`.
- O registro acontece em :func:`register_configs`, chamado uma vez no entrypoint
  (``main.py``, FASE 6) antes de ``@hydra.main``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from src.logger import get_logger

if TYPE_CHECKING:
    import torch

log = get_logger("conf.schema")


# ==============================================================================
# Device — helper modular (auto = MPS ▸ CUDA ▸ CPU) — README §3
# ==============================================================================


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a string de ``device`` da config para um ``torch.device`` concreto.

    Importa ``torch`` *lazy* (o caminho ``random_forest`` não precisa de torch para
    rodar a parte de config). A política de ``"auto"`` é **MPS ▸ CUDA ▸ CPU**, alinhada
    à máquina-alvo (Apple Silicon, sem GPU CUDA).

    Args:
        device: ``"auto"`` | ``"cpu"`` | ``"mps"`` | ``"cuda"``.

    Returns:
        ``torch.device`` resolvido.

    Raises:
        ValueError: Se ``device`` não for um dos valores aceitos.
    """
    import torch

    device = (device or "auto").lower()
    if device == "auto":
        if torch.backends.mps.is_available():
            resolved = "mps"
        elif torch.cuda.is_available():
            resolved = "cuda"
        else:
            resolved = "cpu"
        log.info(f"device=auto resolvido para '{resolved}'")
        return torch.device(resolved)

    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"device inválido: {device!r} (use auto|cpu|mps|cuda)")

    if device == "mps" and not torch.backends.mps.is_available():
        log.warning("device=mps pedido mas MPS indisponível → caindo p/ CPU.")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("device=cuda pedido mas CUDA indisponível → caindo p/ CPU.")
        return torch.device("cpu")
    return torch.device(device)


def device_to_accelerator(device: str | torch.device) -> str:
    """Mapeia o ``device`` para o ``accelerator`` do PyTorch Lightning.

    Mapeamento canônico (README §3): ``cpu→"cpu"`` · ``mps→"mps"`` · ``cuda→"gpu"``.
    Aceita string (``"auto"`` é resolvido antes) ou ``torch.device``. Também exporta
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` quando o alvo é MPS (fallback p/ CPU em ops que o
    Metal não suporta).

    Args:
        device: String de device (já resolvida, não ``"auto"``) ou ``torch.device``.

    Returns:
        ``"cpu"`` | ``"mps"`` | ``"gpu"`` (string aceita por ``L.Trainer``).
    """
    kind = device.type if hasattr(device, "type") else str(device)
    mapping = {"cpu": "cpu", "mps": "mps", "cuda": "gpu"}
    accelerator = mapping.get(kind, "cpu")
    if accelerator == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return accelerator


def seed_everything(seed: int = 42) -> int:
    """Fixa a seed de ``random``, ``numpy`` e (lazy) ``torch`` para reprodutibilidade.

    Substitui ``L.seed_everything`` (que importaria Lightning no caminho ``random_forest``):
    semeia a stdlib e o NumPy sempre, e o ``torch`` (CPU/MPS/CUDA) só se já estiver
    instalado/importável — mantendo o caminho RF 100% sem torch. Chamado uma vez no
    entrypoint (``main.py``, FASE 6) antes de qualquer trabalho.

    Args:
        seed: Semente inteira (default 42; vem de ``cfg.seed``).

    Returns:
        A própria ``seed`` (conveniência p/ logging).
    """
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        log.debug("torch ausente — seed_everything semeou só random/numpy (caminho RF).")
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# ==============================================================================
# Schemas por grupo (espelham configs/<grupo>/*.yaml — README §7)
# ==============================================================================


@dataclass
class WindowConfig:
    """Sub-bloco ``data.window`` — janela deslizante + rotulagem por sobreposição.

    O rótulo de cada janela vem de ``label_source`` (``time_detailed_ah``): a janela
    recebe ``1`` se a fração sobreposta com algum intervalo de A/H for
    ``>= min_overlap_for_positive`` (README §2/§6.1).
    """

    size_s: float = 5.0  # ≈ duração média de A/H (4,3 s)
    hop_s: float = 2.5  # 50% de sobreposição
    min_overlap_for_positive: float = 0.5
    pad_last: bool = True
    label_source: Literal["time_detailed_ah"] = "time_detailed_ah"


@dataclass
class PathsConfig:
    """Sub-bloco ``data.paths`` — caminhos do dataset BAH + artefatos derivados.

    ``data_root`` reflete o layout REAL em disco (pasta ``data/`` aninhada,
    ``data/raw/data``) descoberto na branch ``matheus`` (README §2).
    """

    data_root: str = "data/raw/data"
    videos_dir: str = "data/raw/data/Videos"
    transcription_dir: str = "data/raw/data/transcription"
    split_dir: str = "data/raw/data/split"
    annotation_yaml: str = "data/raw/data/video_annotation_transcript.yaml"
    video_index_csv: str = "data/raw/data/bah-video.csv"
    audio_dir: str = "data/interim/Audio"
    interim_dir: str = "data/interim"
    window_index: str = "data/interim/windows.parquet"
    processed_dir: str = "data/processed"
    parquet_path: str = "data/processed/text_audio_windows.parquet"
    output_root: str = "outputs"


@dataclass
class AudioConfig:
    """Sub-bloco ``data.audio`` — SR canônico do pipeline (extração + janelamento)."""

    sample_rate: int = 16000
    mono: bool = True
    extract_backend: Literal["ffmpeg", "torchaudio"] = "ffmpeg"


@dataclass
class HesitationConfig:
    """Config do :class:`~src.features.hesitation.HesitationExtractor` (marcadores de incerteza).

    Bloco acústico hand-crafted de INCERTEZA/HESITÂNCIA (latência + pausas + entonação
    + volume + jitter/shimmer), alinhado ao construto do BAH — NÃO disfluência clínica.
    Serve de SUPORTE ao embedding. Usado tanto em ``data.tabular.hesitation`` (acompanha
    qualquer backend) quanto em ``audio_embedder.hesitation`` (token ``"hesitation"`` no
    ``feature_set``).
    """

    frame_length: int = 2048
    hop_length: int = 512
    silence_rms_threshold: float = 0.1  # fração do RMS máx. p/ frame de silêncio
    top_db: float = 30.0  # VAD de pausas (dB abaixo do pico) — librosa.effects.split
    min_pause_s: float = 0.15  # duração mínima p/ contar uma pausa
    long_pause_s: float = 0.5  # limiar de pausa "longa" (marcador de incerteza)
    f0_min: float = 65.0  # faixa de F0 do Praat (Hz)
    f0_max: float = 500.0
    final_frac: float = 0.30  # fração final dos frames vozeados p/ a subida terminal de F0
    nuclei_silence_db: float = 25.0  # piso de silêncio (dB abaixo do pico) p/ núcleos silábicos
    nuclei_min_dip_db: float = 2.0  # vale mínimo (dB) entre núcleos — de Jong & Wempe
    use_praat: bool = True  # F0/jitter/shimmer via parselmouth (Praat)


@dataclass
class TextFeaturesConfig:
    """Config do :class:`~src.features.text_features.TextFeaturizer` (ambivalência/hesitância).

    Bloco de texto hand-crafted alinhado ao construto do BAH: A1 (índices psicométricos de
    ambivalência, NÍVEL DE RESPOSTA), A3 (entropia por janela + flutuação de emoção por
    resposta), A4 (contraste + shift de polaridade, por janela), H1 (hedges, por janela).
    Feature de SUPORTE ao embedding de texto. Usado em ``data.tabular.text_features`` (late
    fusion) e ``text_embedder.text_features`` (early fusion). Léxicos são constantes editáveis
    em ``text_features.py``; aqui ficam os flags e parâmetros.
    """

    use_a1: bool = True  # índices de ambivalência (resposta)
    use_a3: bool = True  # entropia (janela) + flutuação/conflito de emoção (resposta)
    use_a4: bool = True  # contraste + shift de polaridade (janela)
    use_h1: bool = True  # hedges (janela)
    use_emotion: bool = True  # carrega o classificador de emoção (P/N emoção em A1 + A3)
    use_hedge_classifier: bool = False  # H1 contextual (logreg de supervisão distante)
    emotion_model: str = "cardiffnlp/twitter-roberta-base-emotion"
    positive_labels: list[str] = field(default_factory=lambda: ["joy", "optimism"])
    negative_labels: list[str] = field(default_factory=lambda: ["anger", "sadness"])
    max_length: int = 128
    batch_size: int = 32
    norm: list[str] = field(default_factory=lambda: ["per_word", "per_100", "raw"])


@dataclass
class TabularConfig:
    """Sub-bloco ``data.tabular`` — grupos de features tabulares habilitados (FASE 3)."""

    use_metadata: bool = True
    use_question_type: bool = True
    use_prosody: bool = True
    use_hesitation: bool = False  # anexa o bloco de hesitação ao lado do embedding
    hesitation: HesitationConfig = field(default_factory=HesitationConfig)
    use_text_features: bool = False  # bloco de texto (ambivalência/hesitância) — late fusion
    text_features: TextFeaturesConfig = field(default_factory=TextFeaturesConfig)


@dataclass
class DataConfig:
    """Grupo ``data`` — paths, áudio, janelamento, tabular e cache Parquet.

    Sub-blocos aninhados, consumidos como ``cfg.data.paths.*``, ``cfg.data.audio.*``,
    ``cfg.data.window.*`` e ``cfg.data.tabular`` pelas FASES 2/3/6.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    tabular: TabularConfig = field(default_factory=TabularConfig)
    batch_size: int = 32
    num_workers: int = 0


@dataclass
class TextEmbedderConfig:
    """Grupo ``text_embedder`` — encoder de texto HuggingFace (agnóstico ao modelo).

    Default = ``cardiffnlp/twitter-roberta-base-emotion`` (RoBERTa EN, emoção; README §8).
    Trocar de modelo é só selecionar outro arquivo do grupo (``text_embedder=minilm``).
    """

    name: str = "roberta_emotion"
    model_name: str = "cardiffnlp/twitter-roberta-base-emotion"
    pooling: Literal["mean", "cls"] = "mean"
    max_length: int = 128
    batch_size: int = 32
    dim: int = 768
    trust_remote_code: bool = False  # exigido por encoders com código custom (ex.: GTE-Large v1.5)
    # Early fusion opcional: embute o TextFeaturizer NO text_emb (passa pela atenção do cross-attn).
    fuse_text_features: bool = False
    text_features: TextFeaturesConfig = field(default_factory=TextFeaturesConfig)


@dataclass
class AudioEmbedderConfig:
    """Grupo ``audio_embedder`` — factory por ``backend`` (README §6.3).

    ``librosa`` (prosódia, CPU) | ``wav2vec2`` | ``hubert`` (deep, respeitam ``device``).
    Campos específicos de cada backend convivem; a FASE 3 lê só os relevantes.
    """

    name: str = "librosa"
    backend: Literal["librosa", "wav2vec2", "hubert"] = "librosa"
    sample_rate: int = 16000
    # librosa
    feature_set: list[str] = field(
        default_factory=lambda: ["mfcc", "delta", "spectral", "chroma", "zcr", "rms", "f0", "tempo"]
    )
    n_mfcc: int = 20
    agg_stats: list[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    # bloco de hesitação (ativo se "hesitation" ∈ feature_set) — ver HesitationConfig
    hesitation: HesitationConfig = field(default_factory=HesitationConfig)
    # deep (wav2vec2/hubert)
    model_name: str | None = None
    pooling: Literal["mean", "cls"] = "mean"
    dim: int | None = None  # derivado em runtime (FASE 3)


@dataclass
class ModelConfig:
    """Grupo ``model`` — registra a família (``sklearn`` | ``lightning``) + hiperparâmetros.

    ``family`` decide o trainer (README §6.4): ``random_forest`` → ``SklearnTrainer``;
    ``cross_attention`` → ``LightningTrainer`` (importado *lazy*). Campos não usados por
    uma família são simplesmente ignorados por ela.
    """

    name: str = "random_forest"
    family: Literal["sklearn", "lightning"] = "sklearn"
    # --- random_forest (sklearn) ---
    n_estimators: int = 400
    max_depth: int | None = None
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: str = "balanced"
    n_jobs: int = -1
    random_state: int = 42
    # --- cross_attention (lightning) ---
    dim_a: int = 768
    dim_b: int = 768
    dim_tab: int | None = None  # inferido do cache no fit (ramo tabular opcional)
    use_tabular: bool = False  # funde tab_seq (hesitação + tabulares) na cross-attention
    common_dim: int = 512
    num_heads: int = 4
    num_classes: int = 1
    dropout: float = 0.1
    lr: float = 1e-3


@dataclass
class TrainerConfig:
    """Grupo ``trainer`` — protocolo de treino das duas famílias.

    Caminho ``sklearn`` (CPU, sem Lightning) usa só ``calibrate_threshold``/``metric``.
    Caminho ``lightning`` deriva ``accelerator`` de ``device`` (README §3).
    """

    family: Literal["sklearn", "lightning"] = "sklearn"
    # sklearn
    calibrate_threshold: bool = True
    metric: str = "macro_f1"
    # lightning
    max_epochs: int = 200
    accelerator: str = "auto"
    devices: int = 1
    gradient_clip_val: float = 1.0
    patience: int = 20
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class AggregationConfig:
    """Grupo ``aggregation`` — janela → vídeo + calibração de limiar (README §6.5)."""

    method: Literal["mean_proba", "max_proba", "frac_positive", "any"] = "mean_proba"
    threshold: float | str = "auto"  # "auto" = calibrado na val; ou float fixo
    calibration: Literal["smooth", "argmax"] = "smooth"  # robustez na escolha do limiar
    smooth_window: float = 0.10  # largura da média móvel (unidades de limiar) p/ 'smooth'
    metric: str = "macro_f1"


@dataclass
class WandbConfig:
    """Sub-bloco ``wandb`` — tracking primário (FASE 5)."""

    project: str = "abaw-ah"
    mode: Literal["online", "offline", "disabled"] = "online"
    group: str | None = None


# ------------------------------------------------------------------------------
# Container raiz (espelha configs/config.yaml)
# ------------------------------------------------------------------------------


@dataclass
class RootConfig:
    """Container raiz da configuração Hydra do experimento BAH (espelha ``config.yaml``).

    Composto pela *defaults list*; cada campo abaixo recebe o grupo selecionado. Os
    campos planos (``seed``/``device``/``mode``/``wandb``) vêm do próprio ``config.yaml``
    (``_self_``) e podem ser sobrescritos por CLI.
    """

    data: DataConfig = field(default_factory=DataConfig)
    text_embedder: TextEmbedderConfig = field(default_factory=TextEmbedderConfig)
    audio_embedder: AudioEmbedderConfig = field(default_factory=AudioEmbedderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    seed: int = 42
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    mode: Literal["train", "preprocess", "featurize", "evaluate", "submit"] = "train"
    experiment_name: str = "abaw-ah"

    # Overrides opcionais por modo (evaluate/submit). None = default do handler.
    split: str | None = None
    out: str | None = None
    checkpoint: str | None = None


# ==============================================================================
# Registro no ConfigStore
# ==============================================================================


def register_configs() -> None:
    """Registra os schemas no ``ConfigStore`` (validação/IDE).

    Chamado uma única vez no entrypoint (``main.py``, FASE 6) antes de ``@hydra.main``.
    Registra o schema raiz com ``name="base_config"`` (referenciado no topo da defaults
    list do ``config.yaml`` se quisermos validação estrita) e cada grupo sob seu nó.
    """
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=RootConfig)
    cs.store(group="data", name="schema", node=DataConfig)
    cs.store(group="text_embedder", name="schema", node=TextEmbedderConfig)
    cs.store(group="audio_embedder", name="schema", node=AudioEmbedderConfig)
    cs.store(group="model", name="schema", node=ModelConfig)
    cs.store(group="trainer", name="schema", node=TrainerConfig)
    cs.store(group="aggregation", name="schema", node=AggregationConfig)
    log.debug("Schemas registrados no ConfigStore.")


def to_container(cfg: Any) -> dict[str, Any]:
    """Converte um ``DictConfig`` (OmegaConf) em ``dict`` puro (snapshot/log/W&B).

    Resolve interpolações (``${...}``) para que o snapshot salvo seja autossuficiente.

    Args:
        cfg: ``DictConfig`` recebido em ``@hydra.main``.

    Returns:
        ``dict`` Python plano, pronto p/ ``json.dump`` ou ``wandb.config``.
    """
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
