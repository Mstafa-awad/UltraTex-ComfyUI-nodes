"""Content-addressed disk arrays and one aggregate, bounded RAM cache."""
from __future__ import annotations

import hashlib
import gc
import json
import os
import shutil
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .contracts import ArrayRef, RuntimePolicy

_LOCK = threading.RLock()
_CACHE = OrderedDict()
_CACHE_BYTES = 0
_PROCESS_ID = uuid.uuid4().hex
_OBJECT_IDS = weakref.WeakKeyDictionary()


def fingerprint(*parts):
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            a = np.ascontiguousarray(part)
            h.update(str(a.shape).encode())
            h.update(a.dtype.str.encode())
            h.update(memoryview(a).cast("B"))
        elif isinstance(part, ArrayRef):
            h.update(part.digest.encode())
        else:
            h.update(json.dumps(part, sort_keys=True, default=str).encode())
        h.update(b"\x00")
    return h.hexdigest()


def object_identity(obj):
    with _LOCK:
        try:
            token = _OBJECT_IDS.get(obj)
            if token is None:
                token = f"{_PROCESS_ID}:{type(obj).__name__}:{uuid.uuid4().hex}"
                _OBJECT_IDS[obj] = token
            return token
        except TypeError:
            # Real model modules support weak references. Primitive test sentinels may not.
            return f"{_PROCESS_ID}:{type(obj).__name__}:{id(obj)}"


def interrupt():
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.throw_exception_if_processing_interrupted()


def ram_preflight(required=0, headroom_mb=2048):
    """Reclaim this pack's cache under pressure, then attempt real allocations.

    Estimates are cache-trimming hints only; they never reject an operation.
    Actual allocator errors propagate with their original diagnostic.
    """
    global _CACHE_BYTES
    try:
        import psutil
    except ImportError:
        return
    required = max(0, int(required))
    target = max(0, headroom_mb) * 2**20
    available = psutil.virtual_memory().available
    if available < required + target:
        with _LOCK:
            _CACHE.clear()
            _CACHE_BYTES = 0
        gc.collect()
        available = psutil.virtual_memory().available
    return {"available_bytes": available, "estimated_bytes": required,
            "policy": "attempt_actual_allocation"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


SESSION_PREFIX = "session_"
_SESSION_DIR = None


def _process_token(pid=None):
    """'<pid>_<creation time>': stays unique when Windows reuses a PID."""
    try:
        import psutil
        process = psutil.Process(pid) if pid is not None else psutil.Process()
        return f"{process.pid}_{int(process.create_time())}"
    except Exception:
        return f"{pid if pid is not None else os.getpid()}_0"


def _remove_tree(path):
    shutil.rmtree(path, ignore_errors=True)


def default_spill_directory(output_dir):
    """This ComfyUI process's own cache folder: <output>/.ultratex_cache/session_<pid>_<start>.

    Intermediates are reused while ComfyUI runs and deleted when it exits; a folder left behind by a crash or a
    closed console window is deleted at the next start (``remove_stale_sessions``)."""
    global _SESSION_DIR
    with _LOCK:
        if _SESSION_DIR is None:
            import atexit
            _SESSION_DIR = Path(output_dir).resolve() / ".ultratex_cache" / (SESSION_PREFIX + _process_token())
            atexit.register(_remove_tree, _SESSION_DIR)
        return _SESSION_DIR


def remove_stale_sessions(output_dir):
    """Delete cache folders of ComfyUI processes that are no longer running, and loose files left by older
    UltraTex versions (flat .npy/.json cache). A folder of a live ComfyUI process is never touched."""
    root = Path(output_dir).resolve() / ".ultratex_cache"
    if not root.is_dir():
        return {"removed": 0}
    removed = 0
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.startswith(SESSION_PREFIX):
            pid_text = entry.name[len(SESSION_PREFIX):].split("_", 1)[0]
            if pid_text.isdigit() and entry.name == SESSION_PREFIX + _process_token(int(pid_text)):
                continue  # owner still running
            _remove_tree(entry)
            removed += 1
        elif entry.is_file() and entry.suffix in (".npy", ".json", ".tmp"):
            entry.unlink(missing_ok=True)
            removed += 1
    return {"removed": removed}


class Store:
    """Content/key-addressed arrays on disk. One ``<token>.npy`` per entry, no side files.

    The token is sha256 of the key (unkeyed arrays use their content fingerprint as key), and it doubles as the
    ArrayRef digest, so identities stay a pure function of the inputs. Files are written to a temporary name and
    renamed, so a present file is always complete."""

    def __init__(self, policy: RuntimePolicy):
        self.policy = policy
        self.root = Path(policy.spill_directory).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.writes = 0

    def _path(self, key):
        token = hashlib.sha256(key.encode()).hexdigest()
        return self.root / (token + ".npy"), token

    def lookup(self, key):
        path, token = self._path(key)
        if not path.is_file():
            return None
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            ref = ArrayRef(str(path), tuple(array.shape), array.dtype.str, token)
            del array
            self.hits += 1
            return ref
        except (ValueError, OSError, EOFError):
            return None

    def put(self, value, key=None):
        array = np.ascontiguousarray(value)
        key = key or fingerprint(array)
        with _LOCK:
            old = self.lookup(key)
            if old is not None and old.shape == tuple(array.shape) and old.dtype == array.dtype.str:
                return old
            path, token = self._path(key)
            used = sum(p.stat().st_size for p in self.root.glob("*.npy"))
            if used + array.nbytes + 512 > self.policy.disk_budget_gb * 2**30:
                raise MemoryError("UltraTex spill budget reached. Increase disk_budget_gb or "
                                  "clear the cache when no workflow references it; live arrays are never evicted.")
            if shutil.disk_usage(self.root).free < array.nbytes + 256 * 2**20:
                raise OSError("UltraTex: insufficient disk space for the next intermediate")
            tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                with tmp.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
            self.writes += 1
            return ArrayRef(str(path), tuple(array.shape), array.dtype.str, token)

    def cached(self, key, factory):
        ref = self.lookup(key)
        return ref if ref is not None else self.put(factory(), key)

    def small(self, ref):
        global _CACHE_BYTES
        limit = self.policy.ram_cache_mb * 2**20
        with _LOCK:
            while _CACHE and _CACHE_BYTES > limit:
                _, old = _CACHE.popitem(last=False)
                _CACHE_BYTES -= old.nbytes
            if ref.digest in _CACHE:
                item = _CACHE.pop(ref.digest)
                _CACHE[ref.digest] = item
                return item
            data = ref.read()
            if data.nbytes > min(limit // 8, 32 * 2**20):
                return data
            value = np.array(data)
            value.flags.writeable = False
            while _CACHE and _CACHE_BYTES + value.nbytes > limit:
                _, old = _CACHE.popitem(last=False)
                _CACHE_BYTES -= old.nbytes
            if value.nbytes <= limit:
                _CACHE[ref.digest] = value
                _CACHE_BYTES += value.nbytes
            return value


@contextmanager
def measure(stage, **details):
    report = {"stage": stage, **details}
    start = time.perf_counter()
    try:
        yield report
    finally:
        report["wall_seconds"] = round(time.perf_counter() - start, 5)


def tensor_from(ref):
    import torch
    # Own a writable, bounded CPU allocation rather than exposing read-only mmap to torch.
    return torch.from_numpy(np.array(ref.read()))
