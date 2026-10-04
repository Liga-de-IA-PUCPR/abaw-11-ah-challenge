"""Garantias do versionamento de ``outputs/`` no bucket do HF (``scripts/outputs_bucket.py``).

Tudo offline: um dublê da ``HfApi`` guarda o bucket em memória com o hash Xet REAL de
cada arquivo (``hf_xet.hash_files``, o mesmo que o servidor devolve em ``xetHash``) e,
como o huggingface_hub, grava o mtime em milissegundos no upload e não o restaura no
download. As garantias testadas são as de NÃO PERDER DADO:

- ``push`` nunca apaga no bucket, arquiva a versão anterior antes de sobrescrever e
  guarda a cópia local mais velha com conteúdo inédito; cada push que muda algo é uma
  versão com manifesto; bucket público é recusado;
- ``pull`` restaura o mtime (o push seguinte não reenvia nada), nunca troca um arquivo
  local cujo conteúdo o bucket não tem e não deixa arquivo truncado se o download falha;
- ``free`` só apaga o que tem o MESMO conteúdo no bucket, nunca o peso de um run em
  treino, e o ``pull`` completo não traz de volta o que ele liberou.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import hf_xet
import pytest
from huggingface_hub import BucketFile
from huggingface_hub.errors import BucketNotFoundError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "outputs_bucket.py"
_spec = importlib.util.spec_from_file_location("outputs_bucket", SCRIPT)
assert _spec is not None and _spec.loader is not None
ob = importlib.util.module_from_spec(_spec)
sys.modules["outputs_bucket"] = ob  # o @dataclass resolve as anotações pelo módulo
_spec.loader.exec_module(ob)

BUCKET = "user/outputs"
OLD = time.time() - 3 * 24 * 3600  # mtime antigo: fora da janela de proteção do free


def _iso_ms(ms: int) -> str:
    """Formato do mtime do servidor (``...%fZ``), o que o huggingface_hub sabe ler."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class FakeBucketApi:
    """Dublê da ``HfApi``: o bucket em memória (conteúdo, hash Xet real, mtime em ms)."""

    def __init__(self, *, exists: bool = True, private: bool = True) -> None:
        self.exists = exists
        self.private = private
        self.objects: dict[str, tuple[bytes, str, int]] = {}
        self.created: list[str] = []
        self.deletes: list[str] = []
        self.fail_paths: set[str] = set()

    # --- carga e inspeção do "bucket" --------------------------------------
    def put(self, rel: str, path: Path) -> None:
        """Sobe ``path`` como ``rel`` (hash Xet real, mtime do arquivo em ms)."""
        (info,) = hf_xet.hash_files([str(path)])
        self.objects[rel] = (path.read_bytes(), info.hash, int(path.stat().st_mtime * 1000))

    def data(self, rel: str) -> bytes:
        return self.objects[rel][0]

    def current(self) -> set[str]:
        """Caminhos atuais (fora do arquivo de versões)."""
        return {rel for rel in self.objects if not rel.startswith(".versions/")}

    def versions(self) -> list[dict]:
        """Manifestos gravados, do mais velho para o mais novo."""
        return [
            json.loads(self.data(rel))
            for rel in sorted(self.objects)
            if rel.endswith("/manifest.json")
        ]

    def _file(self, rel: str) -> BucketFile:
        data, xet, mtime_ms = self.objects[rel]
        stamp = _iso_ms(mtime_ms)
        return BucketFile(
            type="file", path=rel, size=len(data), xetHash=xet, mtime=stamp, uploadedAt=stamp
        )

    # --- interface usada pelo script ---------------------------------------
    def whoami(self) -> dict:
        return {"name": "user", "orgs": [], "auth": {"accessToken": {"role": "write"}}}

    def bucket_info(self, bucket_id: str) -> SimpleNamespace:
        if not self.exists:
            response = SimpleNamespace(headers={}, request=None)
            raise BucketNotFoundError(f"{bucket_id} não existe", response=response)  # type: ignore[arg-type]
        size = sum(len(data) for data, _, _ in self.objects.values())
        return SimpleNamespace(private=self.private, total_files=len(self.objects), size=size)

    def create_bucket(self, bucket_id: str, *, private: bool, exist_ok: bool) -> None:
        self.created.append(bucket_id)
        self.exists, self.private = True, private

    def list_bucket_tree(self, bucket_id: str, prefix=None, *, recursive=None):
        for rel in sorted(self.objects):
            if prefix is None or rel.startswith(prefix):  # prefixo por STRING, como o servidor
                yield self._file(rel)

    def get_bucket_paths_info(self, bucket_id: str, paths) -> list[BucketFile]:
        return [self._file(rel) for rel in paths if rel in self.objects]

    def batch_bucket_files(self, bucket_id: str, *, add=None, copy=None, delete=None) -> None:
        self.deletes += list(delete or [])
        now_ms = int(time.time() * 1000)
        for _, _, xet, dest in copy or []:  # cópia no servidor, pelo hash
            data = next(d for d, h, _ in self.objects.values() if h == xet)
            self.objects[dest] = (data, xet, now_ms)
        for source, dest in add or []:
            if isinstance(source, bytes):
                self.objects[dest] = (source, hashlib.sha256(source).hexdigest(), now_ms)
            else:
                self.put(dest, Path(source))

    def download_bucket_files(self, bucket_id: str, files) -> None:
        for remote, dest in files:
            data = self.data(remote.path)
            if remote.path in self.fail_paths:
                Path(dest).write_bytes(data[: len(data) // 2])  # escrita parcial
                raise ConnectionError("rede caiu")
            Path(dest).write_bytes(data)  # sem restaurar o mtime, como o huggingface_hub


def _write(root: Path, rel: str, data: bytes | str, *, mtime: float = OLD) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    os.utime(path, (mtime, mtime))
    return path


def run(api: FakeBucketApi, root: Path, *argv: str) -> int:
    """Roda um comando do script contra o dublê (sem o ``whoami`` do ``main``)."""
    args = ob.parse_args(["--bucket", BUCKET, *argv])
    args.hf_user = "tester"
    return ob.COMMANDS[args.command](args, api, root)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = (tmp_path / "outputs").resolve()
    path.mkdir()
    return path


def test_push_creates_private_bucket_and_records_one_version_per_change(root: Path) -> None:
    api = FakeBucketApi(exists=False)
    run_dir = "cross_attention/20260101_000000"
    _write(root, f"{run_dir}/trainer_state.json", "{}")
    _write(root, f"{run_dir}/checkpoints/best.ckpt", os.urandom(2048))
    _write(root, ".DS_Store", "lixo do Finder")
    (root / "wandb").mkdir()
    (root / "wandb" / "latest-run").symlink_to(root / run_dir)  # symlink: nunca sobe

    assert run(api, root, "push") == 0
    assert api.created == [BUCKET] and api.private
    assert api.current() == {f"{run_dir}/trainer_state.json", f"{run_dir}/checkpoints/best.ckpt"}
    (version,) = api.versions()
    assert version["status"] == "complete" and version["hf_user"] == "tester"
    assert version["summary"]["added"] == 2
    assert {entry["action"] for entry in version["files"]} == {"added"}
    assert set(version["git"]) == {"commit", "branch", "dirty"}  # de qual código o push saiu

    assert run(api, root, "push") == 0  # nada mudou: nenhuma versão nova
    assert len(api.versions()) == 1


def test_push_archives_the_previous_version_and_never_deletes(root: Path) -> None:
    api = FakeBucketApi()
    manifest = "cross_attention/ensemble_manifest.txt"
    api.put(manifest, _write(root, manifest, "run_a\n"))
    api.put("old/run.json", _write(root, "old/run.json", "{}"))
    (root / "old/run.json").unlink()  # apagado aqui: continua no bucket
    _write(root, manifest, "run_a\nrun_b\n", mtime=OLD + 60)  # mais novo aqui

    assert run(api, root, "push") == 0
    assert api.data(manifest) == b"run_a\nrun_b\n"
    assert "old/run.json" in api.objects and api.deletes == []
    (version,) = api.versions()
    (entry,) = version["files"]
    assert entry["action"] == "updated"
    archived = entry["previous"]["archived_as"]
    assert archived == f".versions/{version['version']}/archive/{manifest}"
    assert api.data(archived) == b"run_a\n"


def test_push_keeps_an_older_local_copy_the_bucket_does_not_have(root: Path) -> None:
    api = FakeBucketApi()
    rel = "submission/trial-0.txt"
    api.put(rel, _write(root, rel, "bucket, mais novo", mtime=OLD + 60))
    _write(root, rel, "local, mais velho e inédito")

    assert run(api, root, "push") == 0
    assert api.data(rel) == b"bucket, mais novo"  # o atual continua o mais novo
    (version,) = api.versions()
    (entry,) = version["files"]
    assert entry["action"] == "archived_local"
    assert api.data(entry["archived_as"]) == "local, mais velho e inédito".encode()

    assert run(api, root, "push") == 0  # já guardado: não sobe de novo
    assert len(api.versions()) == 1


def test_pull_restores_mtime_so_the_next_push_sends_nothing(root: Path, tmp_path: Path) -> None:
    api = FakeBucketApi()
    other = tmp_path / "outra_maquina"
    for rel in ("ca/r1/trainer_state.json", "ca/r1/eval_test/metrics.json", "ca/r1/empty.txt"):
        api.put(rel, _write(other, rel, b"" if rel.endswith("empty.txt") else os.urandom(64)))

    assert run(api, root, "pull") == 0
    for rel, (data, _, mtime_ms) in api.objects.items():
        assert (root / rel).read_bytes() == data
        assert (root / rel).stat().st_mtime == pytest.approx(mtime_ms / 1000)
    assert not list(root.rglob(f"*{ob.PART_SUFFIX}"))

    assert run(api, root, "push") == 0
    assert api.versions() == []  # nada a enviar: nenhuma versão


def test_pull_never_replaces_unpushed_local_content(root: Path, tmp_path: Path) -> None:
    api = FakeBucketApi()
    rel = "cross_attention/ensemble_manifest.txt"
    api.put(rel, _write(tmp_path / "outra", rel, "da outra máquina", mtime=OLD + 60))
    _write(root, rel, "alteração local não enviada")

    assert run(api, root, "pull") == 0
    assert (root / rel).read_text() == "alteração local não enviada"  # conflito: mantido

    assert run(api, root, "sync") == 0
    assert (root / rel).read_text() == "da outra máquina"  # o mais novo vence...
    (version,) = api.versions()
    (entry,) = version["files"]
    assert api.data(entry["archived_as"]) == "alteração local não enviada".encode()  # ...sem perda


def test_pull_failure_leaves_no_truncated_file(root: Path, tmp_path: Path) -> None:
    api = FakeBucketApi()
    rels = [f"ca/r1/checkpoints/{name}.ckpt" for name in ("a", "b", "c")]
    for rel in rels:
        api.put(rel, _write(tmp_path / "src", rel, os.urandom(1024)))
    api.fail_paths = {rels[1]}

    assert run(api, root, "pull") == 1
    assert not (root / rels[1]).exists()  # nunca um arquivo truncado com o nome final
    assert not list(root.rglob(f"*{ob.PART_SUFFIX}"))
    assert all((root / rel).read_bytes() == api.data(rel) for rel in (rels[0], rels[2]))

    api.fail_paths = set()
    assert run(api, root, "pull") == 0
    assert (root / rels[1]).read_bytes() == api.data(rels[1])


def test_free_evicts_only_hash_verified_weights_and_full_pull_respects_it(
    root: Path, tmp_path: Path
) -> None:
    api = FakeBucketApi()
    done, training = "ca/20260101_000000", "ca/20260102_000000"
    done_ckpt = f"{done}/checkpoints/best.ckpt"
    training_ckpt = f"{training}/checkpoints/best.ckpt"  # run sem trainer_state.json
    stale_ckpt = "rf/20260101_000000/model.joblib"
    _write(root, f"{done}/trainer_state.json", "{}")
    _write(root, "rf/20260101_000000/trainer_state.json", "{}")
    _write(root, done_ckpt, os.urandom(4096))
    _write(root, training_ckpt, os.urandom(4096))
    # Mesmo tamanho, outro conteúdo, mais velho que o do bucket: só o hash denuncia.
    api.put(stale_ckpt, _write(tmp_path / "outra", stale_ckpt, b"a" * 512, mtime=OLD + 60))
    _write(root, stale_ckpt, b"b" * 512)

    assert run(api, root, "free", "--dry-run") == 0
    assert (root / done_ckpt).exists()  # dry-run não apaga

    assert run(api, root, "free") == 0  # push + free
    assert not (root / done_ckpt).exists()
    assert api.data(done_ckpt)  # está no bucket
    assert (root / training_ckpt).exists()  # em treino: protegido (mas já no bucket)
    assert training_ckpt in api.objects
    assert (root / stale_ckpt).read_bytes() == b"b" * 512  # diferente do bucket: mantido
    assert (root / f"{done}/trainer_state.json").exists()  # os leves ficam
    assert set(ob.load_freed(root)) == {done_ckpt}

    assert run(api, root, "pull") == 0  # o pull completo não traz de volta o liberado
    assert not (root / done_ckpt).exists()

    assert run(api, root, "pull", "--prefix", f"outputs/{done}") == 0  # o do run traz
    assert (root / done_ckpt).read_bytes() == api.data(done_ckpt)
    assert ob.load_freed(root) == {}


def test_prefix_respects_directory_boundary(root: Path, tmp_path: Path) -> None:
    api = FakeBucketApi()
    for rel in ("runs/p1/a.json", "runs/p12/b.json"):
        api.put(rel, _write(tmp_path / "src", rel, "{}"))
    assert list(ob.remote_index(api, BUCKET, "runs/p1", ob.EXCLUDE)) == ["runs/p1/a.json"]
    assert ob.normalize_prefix("outputs/runs/p1/", root) == "runs/p1"
    assert ob.normalize_prefix(str(root / "runs"), root) == "runs"
    with pytest.raises(SystemExit):
        ob.normalize_prefix("../data", root)


def test_push_refuses_public_bucket_and_dry_run_creates_nothing(root: Path) -> None:
    _write(root, "r/trainer_state.json", "{}")
    public = FakeBucketApi(private=False)
    with pytest.raises(SystemExit, match="PÚBLICO"):
        run(public, root, "push")
    assert public.objects == {}

    missing = FakeBucketApi(exists=False)
    assert run(missing, root, "push", "--dry-run") == 0
    assert missing.created == [] and missing.objects == {}


def test_log_shows_the_history_of_a_file(root: Path, capsys: pytest.CaptureFixture) -> None:
    api = FakeBucketApi()
    rel = "cross_attention/ensemble_manifest.txt"
    _write(root, rel, "v1")
    run(api, root, "push")
    _write(root, rel, "v2, maior", mtime=OLD + 60)
    run(api, root, "push")
    capsys.readouterr()

    assert run(api, root, "log", "--prefix", rel) == 0
    out = capsys.readouterr().out
    first, second = api.versions()
    assert out.index(second["version"]) < out.index(first["version"])  # a mais nova primeiro
    assert second["files"][0]["previous"]["archived_as"] in out


def test_credentials_refuse_read_token_on_push_and_foreign_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeBucketApi()
    assert ob.check_credentials(api, BUCKET, write=True) == "user"
    read_only = {"name": "user", "orgs": [], "auth": {"accessToken": {"role": "read"}}}
    monkeypatch.setattr(api, "whoami", lambda: read_only)
    with pytest.raises(SystemExit, match="LEITURA"):
        ob.check_credentials(api, BUCKET, write=True)
    assert ob.check_credentials(api, BUCKET, write=False) == "user"  # pull/status/log leem
    with pytest.raises(SystemExit, match="namespace|organização"):
        ob.check_credentials(api, "outra-org/outputs", write=False)
