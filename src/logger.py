"""Logging estruturado para os experimentos do desafio BAH.

Fornece:
- Loggers escopados por módulo (:func:`get_logger`).
- Configuração de console + arquivo (:func:`setup_logging`).
- :class:`ExperimentLogger` — context manager que captura todo o stdout de um run
  para ``training_stdout.log``, mantendo a saída no console.

É o logger usado por TODO o código (``from src.logger import get_logger``). Convive com
o W&B (FASE 5): o W&B cuida das métricas/curvas; este logger cuida do texto/arquivo.
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import TextIO

# ==============================================================================
# Estado Global
# ==============================================================================

_RUN_ID: str | None = None
_LOG_FILE_HANDLER: logging.FileHandler | None = None
_INITIALIZED: bool = False


# ==============================================================================
# Funções principais
# ==============================================================================


def get_run_id() -> str:
    """Obtém (ou gera) o ID do run atual, baseado em timestamp ``YYYYMMDD_HHMMSS``."""
    global _RUN_ID  # noqa: PLW0603
    if _RUN_ID is None:
        _RUN_ID = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return _RUN_ID


def reset_run_id() -> str:
    """Força a geração de um novo ID de run."""
    global _RUN_ID  # noqa: PLW0603
    _RUN_ID = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return _RUN_ID


def setup_logging(
    logs_dir: str | Path = "outputs/logs",
    verbose: bool = False,
    experiment_name: str | None = None,
) -> logging.Logger:
    """Configura o root logger com handlers de console e arquivo.

    Args:
        logs_dir: Diretório dos arquivos de log.
        verbose: Habilita nível DEBUG (default: INFO).
        experiment_name: Prefixo opcional para o nome do arquivo de log.

    Returns:
        Root logger configurado.
    """
    global _LOG_FILE_HANDLER, _INITIALIZED  # noqa: PLW0603

    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # evita handlers duplicados em re-init

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(console)

    # Arquivo (sempre DEBUG)
    run_id = get_run_id()
    log_name = f"{experiment_name}_{run_id}" if experiment_name else run_id
    log_file = logs_path / f"{log_name}.log"
    _LOG_FILE_HANDLER = logging.FileHandler(log_file, encoding="utf-8")
    _LOG_FILE_HANDLER.setLevel(logging.DEBUG)
    _LOG_FILE_HANDLER.setFormatter(logging.Formatter(fmt))
    root.addHandler(_LOG_FILE_HANDLER)

    _INITIALIZED = True
    root.info(f"Logging inicializado. Arquivo: {log_file}")
    return root


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger escopado para um módulo.

    Args:
        name: Nome do logger (tipicamente o nome do módulo, ex.: ``"features.text"``).

    Returns:
        Instância de logger.
    """
    return logging.getLogger(name)


# ==============================================================================
# ExperimentLogger — context manager
# ==============================================================================


class ExperimentLogger:
    """Context manager para logging a nível de experimento.

    Captura todo o stdout para ``training_stdout.log`` mantendo a saída no console.
    O relatório estruturado final e as curvas vão para o Reporter local + W&B (FASE 5).

    Uso:
        from src.conf import to_container
        with ExperimentLogger(output_dir, cfg.model.name):
            run_pipeline(cfg)
    """

    def __init__(self, output_dir: Path | str, experiment_name: str):
        """Inicializa o ExperimentLogger.

        Args:
            output_dir: Diretório para os arquivos de saída.
            experiment_name: Nome para identificação nos logs.
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.log_file: TextIO | None = None
        self.original_stdout = sys.stdout
        self.start_time: datetime | None = None

    def __enter__(self) -> ExperimentLogger:
        """Inicia a captura de saída."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (self.output_dir / "training_stdout.log").open("w", encoding="utf-8")
        self.start_time = datetime.now().astimezone()

        header = (
            f"\n{'=' * 80}\n"
            f"EXPERIMENTO: {self.experiment_name}\n"
            f"INÍCIO: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"OUTPUT DIR: {self.output_dir}\n"
            f"{'=' * 80}\n\n"
        )
        self.log_file.write(header)
        self.log_file.flush()

        sys.stdout = _TeeWriter(self.original_stdout, self.log_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Encerra a captura e faz a limpeza."""
        end_time = datetime.now().astimezone()
        duration = end_time - self.start_time if self.start_time else None

        footer = (
            f"\n\n{'=' * 80}\n"
            f"FIM: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"DURAÇÃO: {duration}\n"
            f"STATUS: {'ERRO' if exc_type else 'SUCESSO'}\n"
            f"{'=' * 80}\n"
        )

        if self.log_file:
            self.log_file.write(footer)
            if exc_type is not None:
                self.log_file.write(f"\nERRO: {exc_type.__name__}: {exc_val}\n")
            self.log_file.close()

        sys.stdout = self.original_stdout

        if exc_type is not None:
            get_logger("experiment").error(f"Experimento falhou: {exc_type.__name__}: {exc_val}")

        return False  # não suprime exceções

    def log(self, message: str) -> None:
        """Escreve diretamente no arquivo de log (sem passar pelo stdout)."""
        if self.log_file:
            ts = datetime.now().astimezone().strftime("%H:%M:%S")
            self.log_file.write(f"[{ts}] {message}\n")
            self.log_file.flush()


class _TeeWriter:
    """Escreve em múltiplos streams simultaneamente (console + arquivo)."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, message: str) -> int:
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


# ==============================================================================
# Utilidades
# ==============================================================================


def log_separator(title: str = "", char: str = "=", width: int = 60) -> None:
    """Imprime um separador visual nos logs."""
    log = get_logger("main")
    if title:
        pad = (width - len(title) - 2) // 2
        log.info(f"{char * pad} {title} {char * pad}")
    else:
        log.info(char * width)


def log_dict(data: dict, name: str = "Config", indent: int = 2) -> None:
    """Loga um dicionário de forma legível (até 1 nível de aninhamento)."""
    log = get_logger("main")
    log.info(f"{name}:")
    for key, value in data.items():
        if isinstance(value, dict):
            log.info(f"{' ' * indent}{key}:")
            for k2, v2 in value.items():
                log.info(f"{' ' * (indent * 2)}{k2}: {v2}")
        else:
            log.info(f"{' ' * indent}{key}: {value}")
