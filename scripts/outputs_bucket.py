# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub>=2.0,<3",
#     "hf-xet>=1.5",
# ]
# ///
"""Versiona ``outputs/`` num Storage Bucket PRIVADO do Hugging Face.

O bucket é o arquivo durável dos artefatos (checkpoints, ``trainer_state.json``,
``eval_*/``, logs do Hydra e do W&B, submissões); o disco é um cache que pode ser
esvaziado (``free``) e reidratado (``pull``). Cada máquina é uma ponta que sobe e
baixa, e as duas direções seguem a mesma regra: **o mtime mais novo vence** e
**nenhum conteúdo se perde**.

- ``push``: local → bucket. Sobe o que é novo ou mais novo aqui e NUNCA apaga nada no
  bucket. Antes de sobrescrever um arquivo, copia a versão anterior (no servidor, pelo
  hash Xet, sem reenviar bytes) para ``.versions/<versão>/archive/<caminho>``; um
  arquivo local mais VELHO que o do bucket, com conteúdo que o bucket não tem, também
  vai para lá. Cada push que muda algo é uma VERSÃO, com o manifesto
  ``.versions/<versão>/manifest.json``: quem (usuário do HF, máquina), quando, de qual
  commit do git, e o que entrou (caminho, tamanho, hash Xet, versão anterior).
- ``pull``: bucket → local. Baixa o que falta aqui ou é mais novo lá, em ``*.hfpart``
  trocado pelo nome final só quando completo, e restaura o mtime do bucket (sem isso, o
  push seguinte reenviaria tudo). Nunca apaga um arquivo local e nunca substitui um
  local cujo conteúdo o bucket não tem: alteração não enviada fica e é listada.
- ``sync``: ``push`` e depois ``pull``, o modo bidirecional.
- ``status``: as duas pontas e o que cada direção faria, sem transferir nada.
- ``free``: ``push`` e depois apaga do disco os pesos (``*.ckpt``, ``*.joblib``, ~98% do
  volume) cujo hash Xet local é IGUAL ao do bucket; nunca os de um run sem
  ``trainer_state.json`` (pode estar treinando). Os liberados ficam anotados em
  ``outputs/.outputs_bucket/freed.json``: o ``pull`` completo não os traz de volta, o
  ``pull --prefix <dir>`` (ou ``--rehydrate``) traz.
- ``log``: o histórico de versões (``--prefix`` filtra um arquivo ou diretório).

A comparação é feita aqui, não pelo ``sync_bucket`` do huggingface_hub, por três
comportamentos dele (``_buckets.py``): transfere quando o TAMANHO difere mesmo que o
destino seja mais novo (uma cópia velha sobrescreveria a nova), não restaura o mtime no
download (o push seguinte reenviaria tudo) e sobrescreve sem guardar a versão anterior
(o bucket não tem versionamento: o que é sobrescrito ou apagado lá some).

O script roda num ambiente PRÓPRIO (metadados PEP 723 acima, travados em
``outputs_bucket.py.lock``): o ``transformers`` do projeto trava ``huggingface_hub<2``,
e antes da 2.0 o ``batch_bucket_files`` ignora as falhas parciais que o servidor
devolve num HTTP 200. Por isso ele não importa ``src``.

Uso (no dia a dia, pelos alvos ``make outputs-*``)::

    uv run scripts/outputs_bucket.py --bucket LF-BF/abaw-11-ah-challenge-outputs status
    uv run scripts/outputs_bucket.py --bucket ... push [--dry-run]
    uv run scripts/outputs_bucket.py --bucket ... pull --prefix cross_attention/20260711_191444
    uv run scripts/outputs_bucket.py --bucket ... sync
    uv run scripts/outputs_bucket.py --bucket ... free [--dry-run]
    uv run scripts/outputs_bucket.py --bucket ... log --prefix cross_attention/ensemble_manifest.txt

Credencial: o token do ``get_token`` do huggingface_hub (``HF_TOKEN``, senão o arquivo do
``hf auth login``). Toda execução imprime o usuário, o papel e a origem do token, e recusa
um token de fora do namespace do bucket, ou só de leitura num push.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import functools
import json
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import hf_xet
import huggingface_hub
from huggingface_hub import BucketFile, HfApi, constants, get_token
from huggingface_hub.errors import BucketNotFoundError, HfHubHTTPError
from huggingface_hub.utils import disable_progress_bars

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Raiz local default: ``data.paths.output_root`` (configs/data/default.yaml), o
#: ``hydra.run.dir`` (configs/config.yaml) e o W&B gravam em ``outputs/`` do repositório.
DEFAULT_LOCAL_ROOT = REPO_ROOT / "outputs"

#: Raiz RESERVADA no bucket: ``<versão>/manifest.json`` (o registro de cada push) e
#: ``<versão>/archive/<caminho>`` (os conteúdos que esse push tirou de "atual").
VERSIONS_ROOT = ".versions"

#: Estado local desta ponta, dentro da raiz local (nunca sobe): o que o ``free`` liberou
#: e os manifestos cujo envio falhou (o próximo push os reenvia).
STATE_DIR = ".outputs_bucket"
FREED_STATE = f"{STATE_DIR}/freed.json"
PENDING_DIR = f"{STATE_DIR}/pending"

#: Sufixo do download em curso (vira o nome final só quando completo).
PART_SUFFIX = ".hfpart"

#: Nunca transferidos, nas duas direções: escrita em curso (``.tmp``), download em curso
#: deste script, lixo do Finder e as raízes reservadas. Padrões ``fnmatch`` sobre o
#: caminho relativo à raiz (``*`` também casa ``/``).
EXCLUDE = (
    "*.DS_Store",
    "*.tmp",
    f"*{PART_SUFFIX}",
    VERSIONS_ROOT,
    f"{VERSIONS_ROOT}/*",
    f"{STATE_DIR}/*",
)

#: O que o ``free`` libera por default: os pesos (``*.ckpt`` do Lightning, ``model.joblib``
#: do RandomForest/LightGBM) são ~98% do volume de ``outputs/``.
FREE_PATTERNS = ("*.ckpt", "*.joblib")

#: Marcador de run CONCLUÍDO: o ``save()`` dos trainers grava o ``trainer_state.json`` no
#: fim do treino (src/training/*_trainer.py). Sem ele, o run pode estar treinando.
RUN_DONE_MARKER = "trainer_state.json"

#: O ``free`` não toca arquivo modificado há menos que isto (pode estar sendo escrito).
FREE_MIN_AGE_S = 15 * 60

#: Tolerância na comparação de mtime, a mesma do ``sync_bucket`` (``_SYNC_TIME_WINDOW_MS``):
#: o bucket guarda o mtime em milissegundos inteiros.
TIME_WINDOW_S = 1.0

#: Lotes do push. Cada lote é um commit no bucket, então um push interrompido preserva os
#: lotes anteriores (o ``batch_bucket_files`` sozinho só divide a cada 1.000 arquivos).
PUSH_BATCH_FILES = 64
PUSH_BATCH_BYTES = 4 * 1024**3

#: Arquivos por lote do pull: um lote que falha perde só os próprios downloads.
PULL_BATCH_FILES = 64

#: Quantas linhas de cada categoria o relatório lista antes de resumir.
LISTING_LIMIT = 15

_BUCKET_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


# =============================================================================
# Utilidades
# =============================================================================


def human(n: float) -> str:
    """Tamanho em bytes → texto legível (base 1024)."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TB"


def matches_any(rel: str, patterns: Iterable[str]) -> bool:
    """``True`` se o caminho relativo casar algum padrão ``fnmatch`` (sensível a caixa)."""
    return any(fnmatch.fnmatchcase(rel, pattern) for pattern in patterns)


def under_prefix(rel: str, prefix: str) -> bool:
    """``True`` se ``rel`` é o próprio ``prefix`` ou está abaixo dele (fronteira de diretório)."""
    return not prefix or rel == prefix or rel.startswith(prefix + "/")


def bucket_uri(bucket_id: str) -> str:
    """URI ``hf://buckets/<namespace>/<nome>`` do bucket."""
    return f"hf://buckets/{bucket_id}"


def display(path: Path) -> str:
    """Caminho para o relatório: relativo ao diretório atual quando possível."""
    with contextlib.suppress(ValueError):
        return str(path.relative_to(Path.cwd())) or "."
    return str(path)


def remote_mtime(item: BucketFile) -> float:
    """Mtime (s) do arquivo no bucket: o do arquivo de origem, ou a data do upload."""
    stamp = item.mtime or item.uploaded_at
    return stamp.timestamp() if stamp is not None else 0.0


def iso(ts: float) -> str:
    """Timestamp (s) → ISO 8601 em UTC."""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="milliseconds")


def normalize_prefix(raw: str, root: Path) -> str:
    """Alvo relativo à raiz local, a partir de ``<dir>``, ``outputs/<dir>`` ou um caminho absoluto.

    Raises:
        SystemExit: Caminho fora da raiz local ou com ``..``.
    """
    text = (raw or "").strip()
    if Path(text).is_absolute():
        try:
            text = Path(text).resolve().relative_to(root).as_posix()
        except ValueError:
            raise SystemExit(f"--prefix fora da raiz local {root}: {raw}") from None
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise SystemExit(f"--prefix não pode conter '..': {raw}")
    if parts and parts[0] == root.name and not (root / parts[0]).exists():
        parts = parts[1:]  # outputs/<dir>, relativo ao repositório (como o RUN_DIR do Makefile)
    return "/".join(parts)


@dataclass(frozen=True)
class LocalFile:
    """Um arquivo local: tamanho (bytes) e mtime (s)."""

    size: int
    mtime: float


def compare(local: LocalFile, item: BucketFile) -> str:
    """Compara as duas pontas de um arquivo pelo mtime (com tolerância) e pelo tamanho.

    Returns:
        ``"local_newer"``, ``"remote_newer"``, ``"identical"`` ou ``"conflict"`` (mesmo
        mtime dentro da tolerância, tamanhos diferentes).
    """
    delta = local.mtime - remote_mtime(item)
    if delta > TIME_WINDOW_S:
        return "local_newer"
    if delta < -TIME_WINDOW_S:
        return "remote_newer"
    return "identical" if local.size == item.size else "conflict"


def same_content(local: LocalFile, local_hash: str | None, item: BucketFile) -> bool:
    """Mesmo conteúdo: mesmo tamanho e mesmo hash Xet (dois vazios são iguais sem hash)."""
    if local.size != item.size:
        return False
    return local.size == 0 or (local_hash is not None and local_hash == item.xet_hash)


def xet_hashes(root: Path, rels: Iterable[str]) -> dict[str, str]:
    """Hash Xet (o mesmo ``xetHash`` que o bucket guarda) de arquivos locais, sem enviar nada.

    Um arquivo ilegível fica de fora, com aviso: quem chama trata a ausência como
    "conteúdo desconhecido", a decisão conservadora.
    """
    rels = list(rels)
    if not rels:
        return {}
    try:
        infos = hf_xet.hash_files([str(root / rel) for rel in rels])
    except Exception as error:  # noqa: BLE001 — um arquivo ilegível derruba o lote: isola
        if len(rels) == 1:
            print(f"aviso: sem hash de {rels[0]} ({error})", file=sys.stderr)
            return {}
        return {rel: h for one in rels for rel, h in xet_hashes(root, [one]).items()}
    return {rel: info.hash for rel, info in zip(rels, infos, strict=True)}


# =============================================================================
# Índices das duas pontas
# =============================================================================


@dataclass
class LocalScan:
    """Arquivos locais sob o alvo, por caminho relativo POSIX, e os symlinks ignorados."""

    files: dict[str, LocalFile] = field(default_factory=dict)
    symlinks: list[str] = field(default_factory=list)


def local_index(root: Path, prefix: str, exclude: Iterable[str]) -> LocalScan:
    """Varre a raiz local sob ``prefix``.

    Symlinks nunca são seguidos nem enviados: os do W&B (``latest-run``, ``debug*.log``)
    duplicam arquivos do próprio run ou apontam para fora de ``outputs/``.

    Args:
        root: Raiz local de outputs.
        prefix: Arquivo ou subdiretório (relativo à raiz); vazio = tudo.
        exclude: Padrões ``fnmatch`` a ignorar.
    """
    exclude = tuple(exclude)
    scan = LocalScan()
    start = root / prefix if prefix else root
    if start.is_symlink():
        scan.symlinks.append(prefix)
        return scan
    if start.is_dir():
        on_error = functools.partial(print, "aviso: diretório ilegível:", file=sys.stderr)
        entries: Iterable[Path] = (
            dirpath / name for dirpath, _, names in start.walk(on_error=on_error) for name in names
        )
    elif start.is_file():
        entries = [start]
    else:
        return scan
    for path in entries:
        rel = path.relative_to(root).as_posix()
        if matches_any(rel, exclude):
            continue
        try:
            st = path.lstat()
        except FileNotFoundError:  # apagado no meio da varredura
            continue
        except OSError as error:
            print(f"aviso: ignorando {rel} (stat falhou: {error})", file=sys.stderr)
            continue
        if stat.S_ISLNK(st.st_mode):
            scan.symlinks.append(rel)
        elif stat.S_ISREG(st.st_mode):
            scan.files[rel] = LocalFile(size=st.st_size, mtime=st.st_mtime)
    return scan


def remote_index(
    api: HfApi, bucket_id: str, prefix: str, exclude: Iterable[str]
) -> dict[str, BucketFile]:
    """Versões ATUAIS do bucket sob ``prefix`` (sem ``.versions/``), por caminho.

    O filtro de prefixo do servidor é por STRING (``cross_attention/2026`` casaria
    ``cross_attention/20260711_...``); aqui só entra o próprio arquivo ou o que está
    abaixo do diretório.
    """
    exclude = tuple(exclude)
    out: dict[str, BucketFile] = {}
    for item in api.list_bucket_tree(bucket_id, prefix=prefix or None, recursive=True):
        if not isinstance(item, BucketFile):
            continue
        if under_prefix(item.path, prefix) and not matches_any(item.path, exclude):
            out[item.path] = item
    return out


def known_hashes(api: HfApi, bucket_id: str) -> set[str]:
    """Hash Xet de TUDO no bucket, atuais e arquivo de versões: o conteúdo já guardado."""
    return {
        item.xet_hash
        for item in api.list_bucket_tree(bucket_id, recursive=True)
        if isinstance(item, BucketFile)
    }


def load_freed(root: Path) -> dict[str, dict]:
    """O que o ``free`` liberou desta raiz (caminho → tamanho, hash, data)."""
    path = root / FREED_STATE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        print(f"aviso: {path} ilegível ({error}); tratando como vazio", file=sys.stderr)
        return {}


def save_freed(root: Path, freed: dict[str, dict]) -> None:
    """Grava o estado do ``free`` (escrita atômica; vazio = remove o arquivo)."""
    path = root / FREED_STATE
    if not freed:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(freed, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# =============================================================================
# Bucket e credencial
# =============================================================================


def create_api() -> HfApi:
    """``HfApi`` autenticada, com erro legível se não houver token.

    Raises:
        SystemExit: Nenhum token salvo (``hf auth login``) nem ``HF_TOKEN``.
    """
    if get_token() is None:
        raise SystemExit(
            "sem token do HF: rode `hf auth login` ou exporte HF_TOKEN (token com escrita)"
        )
    return HfApi()


def token_source() -> str:
    """De onde veio o token, na ordem do ``get_token`` do huggingface_hub."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return "variável HF_TOKEN"
    return f"arquivo {constants.HF_TOKEN_PATH} (hf auth login)"


def check_credentials(api: HfApi, bucket_id: str, *, write: bool) -> str:
    """Imprime QUEM está autenticado e recusa um token que não serve para o bucket.

    Args:
        api: Cliente autenticado.
        bucket_id: ``<namespace>/<nome>``.
        write: ``True`` para as operações que escrevem no bucket.

    Returns:
        O usuário do HF dono do token (vai para o manifesto de cada versão).

    Raises:
        SystemExit: Token inválido, de um usuário fora do namespace do bucket, ou só de
            leitura numa operação de escrita.
    """
    try:
        who = api.whoami()
    except HfHubHTTPError as error:
        raise SystemExit(f"token inválido ou expirado ({token_source()}): {error}") from error
    user = who.get("name", "?")
    orgs = {org.get("name") for org in who.get("orgs", [])}
    token = (who.get("auth") or {}).get("accessToken") or {}
    role = token.get("role", "?")
    print(f"credencial: {user} (token {role}, de {token_source()}) → {bucket_uri(bucket_id)}")
    namespace = bucket_id.split("/", 1)[0]
    if namespace != user and namespace not in orgs:
        raise SystemExit(
            f"o token é de '{user}', que não é '{namespace}' nem membro dessa organização: "
            "troque o token ou o HF_BUCKET"
        )
    if write and role == "read":
        raise SystemExit(
            "o token é só de LEITURA e esta operação escreve no bucket: gere um token write em "
            "https://huggingface.co/settings/tokens"
        )
    if write and role == "fineGrained" and not _fine_grained_writes(token, namespace):
        print(
            f"aviso: o token fine-grained não tem repo.write em '{namespace}'; se o push falhar "
            "com 403, dê essa permissão ao token (ou use um token write)",
            file=sys.stderr,
        )
    return user


def _fine_grained_writes(token: dict, namespace: str) -> bool:
    """``True`` se um token fine-grained escreve em repositórios (e buckets) de ``namespace``."""
    grants = token.get("fineGrained") or {}
    allowed = set(grants.get("global") or [])
    for scope in grants.get("scoped") or []:
        if (scope.get("entity") or {}).get("name") == namespace:
            allowed |= set(scope.get("permissions") or [])
    return any(
        perm == "repo.write" or (perm.startswith("bucket") and "write" in perm) for perm in allowed
    )


def ensure_private_bucket(api: HfApi, bucket_id: str, *, create: bool) -> bool:
    """Garante que o bucket seja PRIVADO: cria se faltar (``create``), recusa se público.

    Returns:
        ``True`` se o bucket existe (ou acabou de ser criado).

    Raises:
        SystemExit: O bucket existe e é público. Checkpoints e predições derivam do BAH,
            cujo EULA proíbe redistribuição.
    """
    try:
        info = api.bucket_info(bucket_id)
    except BucketNotFoundError:
        if not create:
            return False
        api.create_bucket(bucket_id, private=True, exist_ok=True)
        info = api.bucket_info(bucket_id)
        print(f"bucket criado (privado): {bucket_uri(bucket_id)}")
    if not info.private:
        raise SystemExit(
            f"o bucket {bucket_id} é PÚBLICO: recuso sincronizar outputs com ele (EULA do BAH).\n"
            f"  torne-o privado: hf buckets settings {bucket_id} --private"
        )
    return True


# =============================================================================
# Versões (o registro de cada push)
# =============================================================================


def git_info() -> dict[str, object]:
    """Commit, branch e se a árvore tinha mudanças: de qual código o push saiu."""

    def git(*cmd: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *cmd],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    changes = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current") or None,
        "dirty": None if changes is None else bool(changes),
    }


def describe(item: BucketFile) -> dict[str, object]:
    """Tamanho, hash Xet e mtime de uma versão no bucket (para o manifesto)."""
    return {"size": item.size, "xet_hash": item.xet_hash, "mtime": iso(remote_mtime(item))}


@dataclass
class Version:
    """Uma versão do bucket: um push que mudou algo, e o seu manifesto.

    ``files`` tem uma entrada por arquivo que entrou: ``added`` (novo), ``updated``
    (substituiu a versão anterior, arquivada em ``previous.archived_as``) ou
    ``archived_local`` (cópia local mais velha que a do bucket, com conteúdo inédito,
    guardada em ``archived_as``; o atual do bucket não mudou).
    """

    id: str
    bucket: str
    user: str
    root: Path
    prefix: str
    started_at: str
    files: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @classmethod
    def start(cls, bucket: str, user: str, root: Path, prefix: str) -> Version:
        """Nova versão: id ordenável pela data (UTC, ms) + sufixo aleatório (pushes simultâneos)."""
        now = datetime.now(UTC)
        return cls(
            id=f"{now:%Y%m%dT%H%M%S}.{now.microsecond // 1000:03d}Z-{uuid.uuid4().hex[:4]}",
            bucket=bucket,
            user=user,
            root=root,
            prefix=prefix,
            started_at=now.isoformat(timespec="seconds"),
        )

    def archive_path(self, rel: str) -> str:
        """Onde esta versão guarda o conteúdo de ``rel`` que deixou de ser o atual."""
        return f"{VERSIONS_ROOT}/{self.id}/archive/{rel}"

    @property
    def manifest_path(self) -> str:
        """Caminho do manifesto desta versão no bucket."""
        return f"{VERSIONS_ROOT}/{self.id}/manifest.json"

    def manifest(self) -> dict[str, object]:
        """O registro auditável da versão."""
        counts = Counter(entry["action"] for entry in self.files)
        return {
            "format": 1,
            "version": self.id,
            "bucket": self.bucket,
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "partial" if self.failures else "complete",
            "hf_user": self.user,
            "host": socket.gethostname(),
            "local_root": str(self.root),
            "prefix": self.prefix,
            "git": git_info(),
            "tool": {
                "script": "scripts/outputs_bucket.py",
                "huggingface_hub": huggingface_hub.__version__,
            },
            "summary": {
                "added": counts["added"],
                "updated": counts["updated"],
                "archived_local": counts["archived_local"],
                "bytes": sum(entry["size"] for entry in self.files),
                "failures": len(self.failures),
            },
            "files": self.files,
            "failures": self.failures,
        }


def list_manifests(api: HfApi, bucket_id: str) -> list[BucketFile]:
    """Manifestos de todas as versões, da mais velha para a mais nova (o id começa pela data)."""
    return sorted(
        (
            item
            for item in api.list_bucket_tree(bucket_id, prefix=VERSIONS_ROOT, recursive=True)
            if isinstance(item, BucketFile)
            and under_prefix(item.path, VERSIONS_ROOT)
            and item.path.endswith("/manifest.json")
        ),
        key=lambda item: item.path,
    )


def publish_manifest(api: HfApi, bucket_id: str, root: Path, version: Version) -> None:
    """Grava o manifesto no disco (pendente) e o envia, junto com pendentes anteriores."""
    pending = root / PENDING_DIR / f"{version.id}.json"
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps(version.manifest(), indent=2, ensure_ascii=False), "utf-8")
    send_pending_manifests(api, bucket_id, root)


def send_pending_manifests(api: HfApi, bucket_id: str, root: Path) -> None:
    """Envia os manifestos que ficaram no disco (o envio falhou ou o push foi interrompido)."""
    folder = root / PENDING_DIR
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        dest = f"{VERSIONS_ROOT}/{path.stem}/manifest.json"
        try:
            api.batch_bucket_files(bucket_id, add=[(path.read_bytes(), dest)])
        except Exception as error:  # noqa: BLE001 — o manifesto espera o próximo push
            print(
                f"aviso: manifesto {path.stem} não subiu ({error}); fica em {path}", file=sys.stderr
            )
            continue
        path.unlink()
        print(f"  versão {path.stem}: {bucket_uri(bucket_id)}/{dest}")


# =============================================================================
# Planos
# =============================================================================


@dataclass
class Plan:
    """O que uma direção faria (caminhos relativos à raiz).

    Attributes:
        direction: ``"push"`` ou ``"pull"``.
        new: Faltam no destino.
        newer: Mais novos na origem: substituem o destino (no push, a versão do bucket
            vai antes para o arquivo de versões).
        archive_local: (push) Mais velhos aqui, com conteúdo que o bucket não tem: vão para
            o arquivo de versões, e o atual do bucket fica.
        touch: (pull) Mesmo conteúdo com mtime diferente: só o mtime local é alinhado.
        newer_elsewhere: Mais novos no destino: ficam, e a outra direção os leva.
        conflicts: (pull) O local tem conteúdo que o bucket não tem: mantido.
        freed: (pull) Liberados aqui pelo ``free``: ficam só no bucket.
        identical: Quantos já são iguais nas duas pontas.
        sizes: Tamanho (bytes) de cada caminho listado.
    """

    direction: str
    new: list[str] = field(default_factory=list)
    newer: list[str] = field(default_factory=list)
    archive_local: list[str] = field(default_factory=list)
    touch: list[str] = field(default_factory=list)
    newer_elsewhere: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    freed: list[str] = field(default_factory=list)
    identical: int = 0
    sizes: dict[str, int] = field(default_factory=dict)

    @property
    def transfers(self) -> list[tuple[str, str]]:
        """``(caminho, tipo)`` a transferir, os pequenos primeiro (métricas antes dos pesos)."""
        jobs = [
            (rel, kind)
            for kind, rels in (
                ("new", self.new),
                ("newer", self.newer),
                ("archive", self.archive_local),
            )
            for rel in rels
        ]
        return sorted(jobs, key=lambda job: (self.sizes[job[0]], job[0]))

    @property
    def total_bytes(self) -> int:
        """Bytes a transferir."""
        return sum(self.sizes[rel] for rel, _ in self.transfers)


def plan_push(
    root: Path,
    local: dict[str, LocalFile],
    remote: dict[str, BucketFile],
    known: Callable[[], set[str]],
) -> Plan:
    """Plano local → bucket: o mais novo vence, e nada que só um lado tem se perde.

    Args:
        root: Raiz local.
        local: Índice local.
        remote: Versões atuais do bucket (mesmo alvo e exclusões).
        known: Hashes de tudo no bucket (chamado só se houver cópia local mais velha).
    """
    plan = Plan("push")
    same_size_newer: list[str] = []  # mais novo aqui com o mesmo tamanho: o hash decide
    older_here: list[str] = []  # mais novo no bucket: o hash diz se o local é inédito
    for rel, lf in local.items():
        plan.sizes[rel] = lf.size
        item = remote.get(rel)
        verdict = "new" if item is None else compare(lf, item)
        if verdict == "new":
            plan.new.append(rel)
        elif verdict == "identical":
            plan.identical += 1
        elif verdict == "remote_newer":
            older_here.append(rel)
        elif verdict == "local_newer" and lf.size == remote[rel].size:
            same_size_newer.append(rel)
        else:  # mais novo aqui, ou mesmo mtime com tamanhos diferentes (o local é o vivo)
            plan.newer.append(rel)
    hashes = xet_hashes(root, [rel for rel in same_size_newer + older_here if local[rel].size])
    for rel in same_size_newer:
        if same_content(local[rel], hashes.get(rel), remote[rel]):
            plan.identical += 1
        else:
            plan.newer.append(rel)
    for rel in older_here:
        if saved_in_bucket(local[rel], hashes.get(rel), remote[rel], known):
            plan.newer_elsewhere.append(rel)
        else:
            plan.archive_local.append(rel)
    return plan


def saved_in_bucket(
    local: LocalFile, digest: str | None, item: BucketFile, known: Callable[[], set[str]]
) -> bool:
    """``True`` se o conteúdo local já está no bucket: é o atual, está no arquivo, ou é vazio."""
    if not local.size or same_content(local, digest, item):
        return True
    return digest is not None and digest in known()


def plan_pull(
    root: Path,
    local: dict[str, LocalFile],
    remote: dict[str, BucketFile],
    freed: Iterable[str],
    known: Callable[[], set[str]],
    *,
    include_freed: bool,
) -> Plan:
    """Plano bucket → local: o mais novo vence, sem apagar nem perder nada local.

    Args:
        root: Raiz local.
        local: Índice local.
        remote: Versões atuais do bucket (mesmo alvo e exclusões).
        freed: Caminhos que o ``free`` liberou aqui.
        known: Hashes de tudo no bucket (chamado só se houver local mais velho).
        include_freed: Traz também os liberados (pull com alvo, ou ``--rehydrate``).
    """
    plan = Plan("pull")
    freed = set(freed)
    remote_newer: list[str] = []
    for rel, item in remote.items():
        plan.sizes[rel] = item.size
        lf = local.get(rel)
        if lf is None:
            (plan.new if include_freed or rel not in freed else plan.freed).append(rel)
            continue
        verdict = compare(lf, item)
        if verdict == "identical":
            plan.identical += 1
        elif verdict == "local_newer":
            plan.newer_elsewhere.append(rel)
        elif verdict == "conflict":  # o push resolve (o local vence, o do bucket é arquivado)
            plan.conflicts.append(rel)
        else:
            remote_newer.append(rel)
    hashes = xet_hashes(root, [rel for rel in remote_newer if local[rel].size])
    for rel in remote_newer:
        digest = hashes.get(rel)
        if same_content(local[rel], digest, remote[rel]):
            plan.touch.append(rel)
        elif saved_in_bucket(local[rel], digest, remote[rel], known):
            plan.newer.append(rel)  # o conteúdo local já está guardado no bucket
        else:
            plan.conflicts.append(rel)
    return plan


def _listing(title: str, rels: list[str], mark: str, sizes: dict[str, int] | None = None) -> None:
    """Imprime até ``LISTING_LIMIT`` caminhos de uma categoria."""
    if not rels:
        return
    total = f", {human(sum(sizes[rel] for rel in rels))}" if sizes is not None else ""
    print(f"  {title}: {len(rels)}{total}")
    for rel in sorted(rels)[:LISTING_LIMIT]:
        print(f"    {mark} {rel}")
    if len(rels) > LISTING_LIMIT:
        print(f"    ... e mais {len(rels) - LISTING_LIMIT}")


_REASONS = {
    ("push", "new"): ("+", "novo"),
    ("push", "newer"): ("~", "mais novo aqui; a versão do bucket vai para o arquivo"),
    ("push", "archive"): ("a", "mais velho aqui e inédito: vai para o arquivo de versões"),
    ("pull", "new"): ("+", "falta aqui"),
    ("pull", "newer"): ("~", "mais novo no bucket"),
}


def print_plan(plan: Plan, source: str, dest: str, *, dry_run: bool) -> None:
    """Resumo do plano, com a lista (truncada) de cada categoria."""
    verb = "subir" if plan.direction == "push" else "baixar"
    tag = "[dry-run] " if dry_run else ""
    transfers = plan.transfers
    print(f"{tag}{plan.direction}: {source} → {dest}")
    print(f"  a {verb}: {len(transfers)} ({human(plan.total_bytes)}); iguais: {plan.identical}")
    for rel, kind in transfers[:LISTING_LIMIT]:
        mark, reason = _REASONS[plan.direction, kind]
        print(f"    {mark} {rel}  ({reason}, {human(plan.sizes[rel])})")
    if len(transfers) > LISTING_LIMIT:
        print(f"    ... e mais {len(transfers) - LISTING_LIMIT}")
    if plan.direction == "push":
        _listing("mais novos no bucket (o pull os traz)", plan.newer_elsewhere, "=")
        return
    _listing("só o mtime alinhado ao do bucket (mesmo conteúdo)", plan.touch, "·")
    _listing("mais novos aqui (o push os leva)", plan.newer_elsewhere, "=")
    _listing(
        "conflitos: conteúdo local que o bucket não tem (mantidos; rode o push)",
        plan.conflicts,
        "!",
    )
    _listing(
        "liberados aqui pelo free (ficam no bucket; traga com --prefix ou --rehydrate)",
        plan.freed,
        "-",
        plan.sizes,
    )


def report(failures: list[str], direction: str) -> int:
    """Lista as falhas e devolve o código de saída."""
    if not failures:
        print(f"  {direction} ok")
        return 0
    print(f"  {direction}: {len(failures)} falha(s) (o resto foi transferido):")
    for failure in failures:
        print(f"    x {failure}")
    return 1


# =============================================================================
# Execução do push
# =============================================================================


def _batches(jobs: list[tuple[str, str]], sizes: dict[str, int]) -> Iterator[list[tuple[str, str]]]:
    """Agrupa as transferências do push em lotes por número de arquivos e bytes."""
    batch: list[tuple[str, str]] = []
    total = 0
    for job in jobs:
        size = sizes[job[0]]
        if batch and (len(batch) >= PUSH_BATCH_FILES or total + size > PUSH_BATCH_BYTES):
            yield batch
            batch, total = [], 0
        batch.append(job)
        total += size
    if batch:
        yield batch


def _batch_failures(error: HfHubHTTPError) -> list[str] | None:
    """Falhas por arquivo de um ``BucketBatchError`` (hub>=2); ``None`` se for outro erro."""
    failures = getattr(error, "failures", None)
    if not failures:
        return None
    return [f"{failure.get('path')}: {failure.get('error')}" for failure in failures]


def _upload(api: HfApi, bucket_id: str, items: list[tuple[str | Path | bytes, str]]) -> list[str]:
    """Um lote de uploads; se falhar por um arquivo local, isola arquivo a arquivo.

    Returns:
        Falhas (``"<caminho>: <erro>"``). Erros HTTP que não são de um arquivo específico
        (token, rede, cota) sobem como exceção.
    """
    if not items:
        return []
    try:
        api.batch_bucket_files(bucket_id, add=items)
    except HfHubHTTPError as error:
        failures = _batch_failures(error)
        if failures is None:
            raise
        return failures
    except Exception as error:  # noqa: BLE001 — leitura local (EIO, arquivo sumiu): isola
        if len(items) == 1:
            return [f"{items[0][1]}: {error}"]
        return [failure for item in items for failure in _upload(api, bucket_id, [item])]
    return []


def _archive(
    api: HfApi, bucket_id: str, replaced: dict[str, BucketFile], version: Version
) -> set[str]:
    """Copia, no servidor, as versões que o lote vai substituir e confere que foram guardadas.

    Returns:
        Os caminhos cuja versão anterior está no arquivo; só esses podem ser sobrescritos.
    """
    if not replaced:
        return set()
    copies = [
        ("bucket", bucket_id, item.xet_hash, version.archive_path(rel))
        for rel, item in replaced.items()
    ]
    try:
        api.batch_bucket_files(bucket_id, copy=copies)
    except HfHubHTTPError as error:
        if _batch_failures(error) is None:
            raise
    dests = [version.archive_path(rel) for rel in replaced]
    saved = {item.path: item for item in api.get_bucket_paths_info(bucket_id, dests)}
    archived: set[str] = set()
    for rel, item in replaced.items():
        copy = saved.get(version.archive_path(rel))
        if copy is not None and copy.xet_hash == item.xet_hash:
            archived.add(rel)
        else:
            version.failures.append(f"{rel}: a versão anterior não foi arquivada (não sobrescrita)")
    return archived


def _local_stats(root: Path, rels: Iterable[str]) -> dict[str, LocalFile]:
    """Tamanho e mtime ATUAIS dos arquivos que ainda existem (os que sumiram ficam de fora)."""
    stats: dict[str, LocalFile] = {}
    for rel in rels:
        with contextlib.suppress(OSError):
            st = (root / rel).stat()
            if stat.S_ISREG(st.st_mode):
                stats[rel] = LocalFile(size=st.st_size, mtime=st.st_mtime)
    return stats


def _push_batch(
    api: HfApi, bucket_id: str, root: Path, batch: list[tuple[str, str]], version: Version
) -> None:
    """Um lote do push: arquiva o que vai ser substituído, sobe e confere no bucket."""
    stats = _local_stats(root, [rel for rel, _ in batch])
    present = [(rel, kind) for rel, kind in batch if rel in stats]
    if not present:
        return
    # O atual do bucket AGORA: outra máquina pode ter enviado algo desde a listagem.
    current = {
        item.path: item for item in api.get_bucket_paths_info(bucket_id, [r for r, _ in present])
    }
    uploads: list[tuple[str, str, str]] = []  # (caminho, ação, destino no bucket)
    replaced: dict[str, BucketFile] = {}
    for rel, kind in present:
        if kind == "archive":
            uploads.append((rel, "archived_local", version.archive_path(rel)))
            continue
        item = current.get(rel)
        if item is not None and remote_mtime(item) - stats[rel].mtime > TIME_WINDOW_S:
            version.failures.append(
                f"{rel}: ficou mais novo no bucket durante o push (não enviado)"
            )
            continue
        if item is not None:
            replaced[rel] = item
        uploads.append((rel, "added" if item is None else "updated", rel))
    archived = _archive(api, bucket_id, replaced, version)
    uploads = [up for up in uploads if up[0] not in replaced or up[0] in archived]
    failures = _upload(api, bucket_id, [(str(root / rel), dest) for rel, _, dest in uploads])
    version.failures += failures
    # Conferência: o bucket tem o que foi enviado (tamanho e mtime do arquivo local).
    sent = {
        item.path: item
        for item in api.get_bucket_paths_info(bucket_id, [dest for _, _, dest in uploads])
    }
    failed = {failure.split(": ", 1)[0] for failure in failures}
    for rel, action, dest in uploads:
        item, lf = sent.get(dest), stats[rel]
        if (
            item is None
            or item.size != lf.size
            or abs(remote_mtime(item) - lf.mtime) > TIME_WINDOW_S
        ):
            if dest not in failed:
                version.failures.append(f"{dest}: o bucket não confirmou o envio")
            continue
        entry: dict[str, object] = {"path": rel, "action": action, **describe(item)}
        if action == "updated":
            entry["previous"] = {
                **describe(replaced[rel]),
                "archived_as": version.archive_path(rel),
            }
        elif action == "archived_local":
            entry["archived_as"] = dest
            if rel in current:
                entry["current"] = describe(current[rel])
        version.files.append(entry)


def execute_push(api: HfApi, bucket_id: str, root: Path, plan: Plan, version: Version) -> None:
    """Executa o push em lotes (cada lote é um commit no bucket)."""
    jobs = plan.transfers
    done, sent, total = 0, 0, plan.total_bytes
    for batch in _batches(jobs, plan.sizes):
        _push_batch(api, bucket_id, root, batch, version)
        done += len(batch)
        sent += sum(plan.sizes[rel] for rel, _ in batch)
        print(f"  [{done}/{len(jobs)}] {human(sent)} de {human(total)}", flush=True)


# =============================================================================
# Execução do pull
# =============================================================================


def _changed_meanwhile(dest: Path, before: LocalFile | None) -> bool:
    """``True`` se o arquivo local foi criado ou alterado desde o plano (o local vence)."""
    try:
        st = dest.lstat()
    except FileNotFoundError:
        return False
    return before is None or st.st_size != before.size or st.st_mtime != before.mtime


def _finalize(item: BucketFile, part: Path, dest: Path, before: LocalFile | None) -> str | None:
    """Confere o tamanho, troca ``.hfpart`` → nome final e restaura o mtime do bucket.

    Returns:
        ``None`` se o arquivo foi instalado; senão, o motivo de não ter sido.
    """
    if not part.exists():  # o download_bucket_files pula (com warning) o que sumiu do bucket
        return f"{item.path}: não está mais no bucket"
    size = part.stat().st_size
    if size != item.size:
        return f"{item.path}: download truncado ({size} de {item.size} B)"
    if _changed_meanwhile(dest, before):
        return f"{item.path}: mudou aqui durante o pull (o local foi mantido)"
    part.replace(dest)
    stamp = remote_mtime(item)
    os.utime(dest, (stamp, stamp))
    return None


def _download(
    api: HfApi, bucket_id: str, jobs: list[tuple[BucketFile, Path, LocalFile | None]]
) -> tuple[list[str], set[str]]:
    """Um lote de downloads em ``.hfpart``; se o lote falhar, isola arquivo a arquivo.

    Returns:
        ``(falhas, caminhos instalados)``. Nenhum ``.hfpart`` fica para trás.
    """
    parts = [dest.with_name(dest.name + PART_SUFFIX) for _, dest, _ in jobs]
    try:
        try:
            for part in parts:
                part.parent.mkdir(parents=True, exist_ok=True)
            pairs: list[tuple[str | BucketFile, str | Path]] = [
                (item, part) for (item, _, _), part in zip(jobs, parts, strict=True)
            ]
            api.download_bucket_files(bucket_id, pairs)
        except HfHubHTTPError:
            raise
        except Exception as error:  # noqa: BLE001 — disco (permissão, espaço) ou rede: isola
            if len(jobs) == 1:
                return [f"{jobs[0][0].path}: {error}"], set()
            failures: list[str] = []
            done: set[str] = set()
            for job in jobs:
                one_failures, one_done = _download(api, bucket_id, [job])
                failures += one_failures
                done |= one_done
            return failures, done
        failures, done = [], set()
        for (item, dest, before), part in zip(jobs, parts, strict=True):
            problem = _finalize(item, part, dest, before)
            if problem is None:
                done.add(item.path)
            else:
                failures.append(problem)
        return failures, done
    finally:
        for part in parts:
            with contextlib.suppress(OSError):
                part.unlink(missing_ok=True)


def execute_pull(
    api: HfApi,
    bucket_id: str,
    root: Path,
    remote: dict[str, BucketFile],
    local: dict[str, LocalFile],
    plan: Plan,
) -> tuple[list[str], set[str]]:
    """Executa o pull: alinha os mtimes, depois baixa em lotes.

    Returns:
        ``(falhas, caminhos instalados)``.

    Raises:
        SystemExit: Uma chave do bucket que escaparia da raiz local.
    """
    failures: list[str] = []
    for rel in plan.touch:  # mesmo conteúdo: só o mtime, e só se nada mudou desde o plano
        if _changed_meanwhile(root / rel, local.get(rel)):
            failures.append(f"{rel}: mudou aqui durante o pull (o local foi mantido)")
            continue
        stamp = remote_mtime(remote[rel])
        try:
            os.utime(root / rel, (stamp, stamp))
        except OSError as error:
            failures.append(f"{rel}: {error}")
    root = root.resolve()
    jobs: list[tuple[BucketFile, Path, LocalFile | None]] = []
    for rel, _ in plan.transfers:
        dest = (root / rel).resolve()
        if not dest.is_relative_to(root):
            raise SystemExit(f"caminho inseguro no bucket: {rel!r}")
        jobs.append((remote[rel], dest, local.get(rel)))
    done: set[str] = set()
    for start in range(0, len(jobs), PULL_BATCH_FILES):
        chunk = jobs[start : start + PULL_BATCH_FILES]
        chunk_failures, chunk_done = _download(api, bucket_id, chunk)
        failures += chunk_failures
        done |= chunk_done
        print(f"  [{start + len(chunk)}/{len(jobs)}]", flush=True)
    return failures, done


# =============================================================================
# free
# =============================================================================


@dataclass
class FreeReport:
    """Resultado do ``free`` (caminhos relativos)."""

    evicted: list[str] = field(default_factory=list)
    freed_bytes: int = 0
    not_in_bucket: list[str] = field(default_factory=list)
    mismatch: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def run_done(root: Path, rel: str) -> bool:
    """``True`` se o run do peso ``rel`` terminou (``trainer_state.json`` no run dir).

    O run dir é o pai de ``checkpoints/`` (Lightning) ou o diretório do ``model.joblib``.
    """
    parent = PurePosixPath(rel).parent
    run = parent.parent if parent.name == "checkpoints" else parent
    return (root / run / RUN_DONE_MARKER).exists()


def free(
    root: Path,
    local: dict[str, LocalFile],
    remote: dict[str, BucketFile],
    *,
    patterns: Iterable[str] = FREE_PATTERNS,
    dry_run: bool = False,
    min_age_s: float = FREE_MIN_AGE_S,
) -> FreeReport:
    """Apaga do disco o que casa ``patterns`` e tem no bucket o MESMO conteúdo (hash Xet).

    Args:
        root: Raiz local.
        local: Índice local (o alvo do comando).
        remote: Versões atuais do bucket (mesmo alvo).
        patterns: Padrões ``fnmatch`` do que pode ser liberado.
        dry_run: Confere tudo (inclusive os hashes) mas não apaga nada.
        min_age_s: Não toca arquivo modificado há menos que isto.
    """
    patterns = tuple(patterns)
    result = FreeReport()
    now = time.time()
    candidates: list[str] = []
    for rel, lf in sorted(local.items()):
        if not lf.size or not matches_any(rel, patterns):
            continue
        weight = matches_any(rel, FREE_PATTERNS)
        if now - lf.mtime < min_age_s or (weight and not run_done(root, rel)):
            result.protected.append(rel)
        elif rel not in remote:
            result.not_in_bucket.append(rel)
        elif remote[rel].size != lf.size:
            result.mismatch.append(rel)
        else:
            candidates.append(rel)
    hashes = xet_hashes(root, candidates)
    # O que foi recriado aqui (um pull, um treino novo) deixa de constar como liberado.
    state = {rel: v for rel, v in load_freed(root).items() if not (root / rel).exists()}
    for rel in candidates:
        item = remote[rel]
        if hashes.get(rel) != item.xet_hash:
            result.mismatch.append(rel)
            continue
        if not dry_run:
            try:
                (root / rel).unlink()
            except OSError as error:
                result.failures.append(f"{rel}: {error}")
                continue
            state[rel] = {"size": item.size, "xet_hash": item.xet_hash, "freed_at": iso(now)}
        result.evicted.append(rel)
        result.freed_bytes += item.size
    if result.evicted and not dry_run:
        save_freed(root, state)
    return result


def print_free(result: FreeReport, root: Path, *, dry_run: bool) -> None:
    """Resumo do ``free``."""
    verb = "seriam liberados" if dry_run else "liberados"
    print(f"{'[dry-run] ' if dry_run else ''}free em {display(root)}")
    print(f"  {verb}: {len(result.evicted)} arquivos ({human(result.freed_bytes)})")
    _listing("fora do bucket (rode o push antes)", result.not_in_bucket, "?")
    _listing("diferentes da versão do bucket (mantidos)", result.mismatch, "!")
    _listing(
        f"protegidos (run sem {RUN_DONE_MARKER} ou modificados há < {FREE_MIN_AGE_S // 60} min)",
        result.protected,
        "#",
    )
    if result.evicted and not dry_run:
        print("  traga de volta sob demanda: make outputs-pull PREFIX=<dir> (ou RUN_DIR=<run dir>)")


# =============================================================================
# Comandos
# =============================================================================


def _target(args: argparse.Namespace, root: Path) -> tuple[str, tuple[str, ...]]:
    """Prefixo normalizado e padrões de exclusão do comando."""
    return normalize_prefix(args.prefix, root), (*EXCLUDE, *args.exclude)


def cmd_push(args: argparse.Namespace, api: HfApi, root: Path) -> int:
    """``push``: local → bucket (uma nova versão, se algo mudou)."""
    if not root.is_dir():
        raise SystemExit(f"raiz local não existe: {root}")
    exists = ensure_private_bucket(api, args.bucket, create=not args.dry_run)
    if not exists:
        print("(o bucket não existe: o push o cria PRIVADO)")
    prefix, exclude = _target(args, root)
    scan = local_index(root, prefix, exclude)
    remote = remote_index(api, args.bucket, prefix, exclude) if exists else {}
    known = functools.cache(lambda: known_hashes(api, args.bucket))
    plan = plan_push(root, scan.files, remote, known)
    dest = f"{bucket_uri(args.bucket)}/{prefix}".rstrip("/")
    print_plan(plan, display(root / prefix), dest, dry_run=args.dry_run)
    if scan.symlinks:
        print(f"  symlinks ignorados (nunca sobem): {len(scan.symlinks)}")
    if args.dry_run:
        return 0
    send_pending_manifests(api, args.bucket, root)
    if not plan.transfers:
        print("  push ok (nada mudou: nenhuma versão nova)")
        return 0
    version = Version.start(args.bucket, args.hf_user, root, prefix)
    try:
        execute_push(api, args.bucket, root, plan, version)
    finally:  # o que subiu fica registrado mesmo se o push for interrompido
        if version.files or version.failures:
            publish_manifest(api, args.bucket, root, version)
    return report(version.failures, "push")


def cmd_pull(args: argparse.Namespace, api: HfApi, root: Path) -> int:
    """``pull``: bucket → local."""
    if not ensure_private_bucket(api, args.bucket, create=False):
        raise SystemExit(f"o bucket {args.bucket} não existe (o primeiro push o cria)")
    prefix, exclude = _target(args, root)
    start = root / prefix if prefix else root
    if start.is_dir() and not args.dry_run:
        for stale in start.rglob(f"*{PART_SUFFIX}"):  # sobra de um pull interrompido à força
            stale.unlink(missing_ok=True)
    scan = local_index(root, prefix, exclude)
    remote = remote_index(api, args.bucket, prefix, exclude)
    freed = load_freed(root)
    known = functools.cache(lambda: known_hashes(api, args.bucket))
    rehydrate = bool(prefix) or args.rehydrate
    plan = plan_pull(root, scan.files, remote, freed, known, include_freed=rehydrate)
    source = f"{bucket_uri(args.bucket)}/{prefix}".rstrip("/")
    print_plan(plan, source, display(root / prefix), dry_run=args.dry_run)
    if args.dry_run:
        return 0
    root.mkdir(parents=True, exist_ok=True)
    failures, _ = execute_pull(api, args.bucket, root, remote, scan.files, plan)
    still_freed = {rel: v for rel, v in freed.items() if not (root / rel).exists()}
    if len(still_freed) != len(freed):  # o que voltou deixa de constar como liberado
        save_freed(root, still_freed)
    return report(failures, "pull")


def cmd_sync(args: argparse.Namespace, api: HfApi, root: Path) -> int:
    """``sync``: push e depois pull (bidirecional)."""
    pushed = cmd_push(args, api, root)
    if args.dry_run and not ensure_private_bucket(api, args.bucket, create=False):
        print("[dry-run] pull: o bucket ainda não existe, nada a baixar")
        return pushed
    return max(pushed, cmd_pull(args, api, root))


def cmd_free(args: argparse.Namespace, api: HfApi, root: Path) -> int:
    """``free``: push e depois apaga do disco o que o bucket já tem com o mesmo hash."""
    code = 0 if args.no_push else cmd_push(args, api, root)
    if not ensure_private_bucket(api, args.bucket, create=False):
        print("free: o bucket não existe, nada a liberar")
        return code
    prefix, exclude = _target(args, root)
    scan = local_index(root, prefix, exclude)
    remote = remote_index(api, args.bucket, prefix, exclude)
    patterns = tuple(args.pattern or FREE_PATTERNS)
    result = free(root, scan.files, remote, patterns=patterns, dry_run=args.dry_run)
    print_free(result, root / prefix, dry_run=args.dry_run)
    return max(code, report(result.failures, "free") if result.failures else 0)


def cmd_status(args: argparse.Namespace, api: HfApi, root: Path) -> int:
    """``status``: as duas pontas e o que cada direção faria."""
    exists = ensure_private_bucket(api, args.bucket, create=False)
    prefix, exclude = _target(args, root)
    scan = local_index(root, prefix, exclude)
    freed = {r: v for r, v in load_freed(root).items() if under_prefix(r, prefix)}
    local_bytes = sum(f.size for f in scan.files.values())
    print(f"local  {display(root / prefix)}: {len(scan.files)} arquivos, {human(local_bytes)}")
    if scan.symlinks:
        print(f"  symlinks ignorados (nunca sobem): {len(scan.symlinks)}")
    if freed:
        freed_bytes = sum(v.get("size", 0) for v in freed.values())
        print(f"  liberados pelo free (só no bucket): {len(freed)} arquivos, {human(freed_bytes)}")
    if not exists:
        print(f"bucket {args.bucket}: não existe (o primeiro `make outputs-push` o cria, privado)")
        return 0
    info = api.bucket_info(args.bucket)
    versions = len(list_manifests(api, args.bucket))
    print(
        f"bucket {args.bucket} (privado): {info.total_files} arquivos, {human(info.size)} "
        f"(inclui o arquivo de {versions} versões)"
    )
    remote = remote_index(api, args.bucket, prefix, exclude)
    known = functools.cache(lambda: known_hashes(api, args.bucket))
    dest = f"{bucket_uri(args.bucket)}/{prefix}".rstrip("/")
    local_path = display(root / prefix)
    print_plan(plan_push(root, scan.files, remote, known), local_path, dest, dry_run=True)
    pull = plan_pull(root, scan.files, remote, freed, known, include_freed=bool(prefix))
    print_plan(pull, dest, local_path, dry_run=True)
    freeable = [
        lf.size
        for rel, lf in scan.files.items()
        if matches_any(rel, FREE_PATTERNS)
        and run_done(root, rel)
        and rel in remote
        and remote[rel].size == lf.size
    ]
    print(
        f"liberável pelo free: {len(freeable)} arquivos, {human(sum(freeable))} "
        "(o free ainda confere o hash de cada um)"
    )
    return 0


def _version_line(doc: dict) -> str:
    """Uma linha do ``log``: id, data, quem, de qual commit e o que entrou."""
    git = doc.get("git") or {}
    commit = (git.get("commit") or "?")[:8] + ("*" if git.get("dirty") else "")
    branch = f" ({git['branch']})" if git.get("branch") else ""
    summary = doc.get("summary") or {}
    status = (
        "" if doc.get("status") == "complete" else f"  [parcial: {summary.get('failures')} falhas]"
    )
    counts = (
        f"+{summary.get('added', 0)} ~{summary.get('updated', 0)} "
        f"a{summary.get('archived_local', 0)}"
    )
    who = f"{doc.get('hf_user')}@{doc.get('host')}"
    return (
        f"{doc.get('version')}  {who}  git {commit}{branch}  {counts}"
        f"  ({human(summary.get('bytes', 0))}){status}"
    )


def _entry_line(entry: dict) -> str:
    """Um arquivo de uma versão no ``log --prefix``."""
    mark = {"added": "+", "updated": "~", "archived_local": "a"}.get(str(entry.get("action")), "?")
    digest = entry.get("xet_hash", "")[:12]
    line = f"    {mark} {entry['path']}  {human(entry.get('size', 0))}  xet {digest}"
    previous = entry.get("previous") or {}
    archived = previous.get("archived_as") or entry.get("archived_as")
    return f"{line}  (anterior/arquivo: {archived})" if archived else line


def cmd_log(args: argparse.Namespace, api: HfApi, root: Path) -> int:
    """``log``: o histórico de versões, a mais nova primeiro (``--prefix`` filtra)."""
    if not ensure_private_bucket(api, args.bucket, create=False):
        print(f"bucket {args.bucket}: não existe, sem versões")
        return 0
    prefix = normalize_prefix(args.prefix, root)
    manifests = list_manifests(api, args.bucket)
    if not prefix:  # sem filtro, só as últimas: um manifesto por versão a baixar
        manifests = manifests[-args.limit :]
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / f"{i}.json" for i in range(len(manifests))]
        jobs: list[tuple[str | BucketFile, str | Path]] = list(zip(manifests, paths, strict=True))
        api.download_bucket_files(args.bucket, jobs)
        docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    shown = 0
    for doc in reversed(docs):
        entries = [e for e in doc.get("files", []) if under_prefix(e.get("path", ""), prefix)]
        if prefix and not entries:
            continue
        print(_version_line(doc))
        for entry in entries if prefix else []:
            print(_entry_line(entry))
        shown += 1
        if shown >= args.limit:
            break
    if not shown:
        print("nenhuma versão" + (f" mexeu em {prefix}" if prefix else " ainda"))
    else:
        print(
            "versão anterior de um arquivo: hf buckets cp "
            f"{bucket_uri(args.bucket)}/{VERSIONS_ROOT}/<versão>/archive/<caminho> <destino>"
        )
    return 0


COMMANDS = {
    "push": cmd_push,
    "pull": cmd_pull,
    "sync": cmd_sync,
    "status": cmd_status,
    "free": cmd_free,
    "log": cmd_log,
}

#: Comandos que escrevem no bucket (exigem token com escrita).
WRITE_COMMANDS = {"push", "sync", "free"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--bucket",
        default=os.environ.get("HF_BUCKET"),
        help="bucket <namespace>/<nome> (default: $HF_BUCKET)",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--local", type=Path, default=None, help="raiz local (default: outputs/)")
    common.add_argument(
        "--prefix",
        default="",
        help="só este arquivo ou subdiretório, relativo à raiz (aceita outputs/<...>)",
    )
    common.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PADRÃO",
        help="padrão fnmatch a ignorar, relativo à raiz (repetível; ex.: '*.ckpt')",
    )
    dry = argparse.ArgumentParser(add_help=False)
    dry.add_argument("--dry-run", action="store_true", help="só mostra o plano")
    rehydrate = argparse.ArgumentParser(add_help=False)
    rehydrate.add_argument(
        "--rehydrate", action="store_true", help="traz também o que o free liberou"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("push", parents=[common, dry], help="local → bucket (aditivo, nova versão)")
    sub.add_parser("pull", parents=[common, dry, rehydrate], help="bucket → local")
    sub.add_parser("sync", parents=[common, dry, rehydrate], help="push e depois pull")
    sub.add_parser("status", parents=[common], help="o que cada direção faria")
    p_free = sub.add_parser("free", parents=[common, dry], help="push e libera o disco")
    p_free.add_argument(
        "--pattern",
        action="append",
        default=None,
        metavar="PADRÃO",
        help=f"o que liberar (repetível; default: {' '.join(FREE_PATTERNS)})",
    )
    p_free.add_argument("--no-push", action="store_true", help="não roda o push antes")
    p_log = sub.add_parser("log", parents=[common], help="histórico de versões")
    p_log.add_argument("--limit", type=int, default=20, help="quantas versões (default: 20)")
    args = parser.parse_args(argv)
    if not args.bucket:
        parser.error("informe --bucket <namespace>/<nome> (ou exporte HF_BUCKET)")
    if not _BUCKET_ID.match(args.bucket):
        parser.error(f"--bucket deve ser <namespace>/<nome> (recebido '{args.bucket}')")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if int(huggingface_hub.__version__.split(".")[0]) < 2:
        raise SystemExit(
            f"huggingface_hub {huggingface_hub.__version__} < 2.0 não reporta falhas parciais "
            "do bucket: rode `uv run scripts/outputs_bucket.py` (ambiente próprio, PEP 723)"
        )
    if not sys.stderr.isatty():  # log redirecionado: sem barras de progresso com \r
        disable_progress_bars()
    root = (args.local or DEFAULT_LOCAL_ROOT).resolve()
    try:
        api = create_api()
        args.hf_user = check_credentials(api, args.bucket, write=args.command in WRITE_COMMANDS)
        return COMMANDS[args.command](args, api, root)
    except HfHubHTTPError as error:  # rede, cota, permissão: o que já foi enviado fica
        print(f"erro do Hub: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
