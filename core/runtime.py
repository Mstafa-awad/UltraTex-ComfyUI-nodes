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
    """Reclaim this pack's cache under pressure; reserve only a bounded margin.

    Headroom is a cache-trimming target, not an extra multi-GiB allocation. Large
    offloaded models can leave less than that target while a CPU bake still fits.
    A second measurement after collection avoids requiring a manual queue rerun.
    """
    global _CACHE_BYTES
    try:
        import psutil
    except ImportError:
        return
    required = max(0, int(required))
    target = max(0, headroom_mb) * 2**20
    reserve = min(target, max(64 * 2**20, int(required * .1)))
    available = psutil.virtual_memory().available
    if available < required + target:
        with _LOCK:
            _CACHE.clear()
            _CACHE_BYTES = 0
        gc.collect()
        available = psutil.virtual_memory().available
    if available < required + reserve:
        raise MemoryError(f"UltraTex: {available / 2**30:.2f} GiB system RAM available; "
                          f"this stage needs about {required / 2**30:.2f} GiB "
                          f"plus {reserve / 2**20:.0f} MiB margin after releasing its RAM cache. "
                          "Reduce resolution or release inactive models.")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Store:
    def __init__(self, policy: RuntimePolicy):
        self.policy = policy
        self.root = Path(policy.spill_directory).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.writes = 0

    def _paths(self, key):
        token = hashlib.sha256(key.encode()).hexdigest()
        return self.root / (token + ".npy"), self.root / (token + ".json")

    def lookup(self, key):
        path, meta = self._paths(key)
        if not path.is_file() or not meta.is_file():
            return None
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            if data["version"] != 1 or not data["complete"]:
                return None
            ref = ArrayRef(str(path), tuple(data["shape"]), data["dtype"], data["digest"])
            if path.stat().st_size != data["file_bytes"]:
                return None
            ref.read()  # validate the NPY header before accepting a persistent entry
            self.hits += 1
            return ref
        except (ValueError, KeyError, OSError, EOFError):
            return None

    def put(self, value, key=None):
        array = np.ascontiguousarray(value)
        digest = fingerprint(array)
        key = key or digest
        with _LOCK:
            old = self.lookup(key)
            if old is not None and old.digest == digest:
                return old
            path, meta = self._paths(key)
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
                atomic_json(meta, dict(version=1, complete=True, shape=array.shape, dtype=array.dtype.str,
                                       digest=digest, file_bytes=path.stat().st_size))
            finally:
                tmp.unlink(missing_ok=True)
            self.writes += 1
            return ArrayRef(str(path), tuple(array.shape), array.dtype.str, digest)

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
