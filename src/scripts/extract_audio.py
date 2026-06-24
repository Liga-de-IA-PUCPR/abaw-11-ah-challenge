"""CLI de extração de áudio do BAH (mp4 → flac 16 kHz mono).

Reusa a branch `matheus`: varre ``data/raw/data/Videos`` em busca de ``*.mp4`` e extrai a
faixa de áudio para ``data/interim/Audio/Videos`` preservando a estrutura de pastas. A
extração propriamente dita delega para :func:`src.data.audio_io.extract_audio` (ffmpeg
``-vn -ar 16000 -ac 1``, com fallback torchaudio), de modo que esta CLI e o pipeline
Hydra (FASE 6) compartilham a mesma lógica.

Uso (ver README §10):
    python -m src.scripts.extract_audio
    python -m src.scripts.extract_audio \
        --source data/raw/data/Videos --out data/interim/Audio/Videos
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.audio_io import SAMPLE_RATE, extract_audio
from src.logger import get_logger

log = get_logger("scripts.extract_audio")

# Defaults idênticos ao matheus (layout real em disco — README §2).
SOURCE_ROOT = Path("data/raw/data/Videos")
OUTPUT_ROOT = Path("data/interim/Audio/Videos")


def run(
    source_root: Path,
    output_root: Path,
    sample_rate: int = SAMPLE_RATE,
    mono: bool = True,
    backend: str = "ffmpeg",
    overwrite: bool = False,
) -> int:
    """Extrai o áudio de todos os ``*.mp4`` sob ``source_root`` para ``output_root``.

    Espelha a saída preservando os caminhos relativos (``<rel>.flac``). Pula arquivos
    já extraídos quando ``overwrite=False``.

    Returns:
        Número de arquivos processados (extraídos ou já em cache).
    """
    mp4_files = sorted(source_root.rglob("*.mp4"))
    log.info(f"Encontrados {len(mp4_files)} arquivos MP4 em {source_root}")

    processed = 0
    for mp4_path in mp4_files:
        relative = mp4_path.relative_to(source_root)
        flac_path = (output_root / relative).with_suffix(".flac")
        try:
            extract_audio(
                mp4_path,
                flac_path,
                sample_rate=sample_rate,
                mono=mono,
                backend=backend,
                overwrite=overwrite,
            )
            processed += 1
        except Exception as exc:  # noqa: BLE001 — loga e segue para o próximo arquivo
            log.error(f"Falha ao extrair {relative}: {exc}")

    log.info(f"Extração concluída: {processed}/{len(mp4_files)} arquivos.")
    return processed


def _build_parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos da CLI."""
    parser = argparse.ArgumentParser(
        description="Extrai áudio (flac 16 kHz mono) dos vídeos do BAH."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_ROOT,
        help=f"Pasta raiz dos .mp4 (default: {SOURCE_ROOT}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Pasta raiz de saída dos .flac (default: {OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help=f"Taxa de amostragem-alvo em Hz (default: {SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        help="Mantém 2 canais (default: mono).",
    )
    parser.add_argument(
        "--backend",
        choices=["ffmpeg", "torchaudio"],
        default="ffmpeg",
        help="Backend de extração (default: ffmpeg).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extrai mesmo que o .flac já exista.",
    )
    return parser


def main() -> None:
    """Ponto de entrada da CLI ``python -m src.scripts.extract_audio``."""
    args = _build_parser().parse_args()
    run(
        source_root=args.source,
        output_root=args.out,
        sample_rate=args.sample_rate,
        mono=not args.stereo,
        backend=args.backend,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
