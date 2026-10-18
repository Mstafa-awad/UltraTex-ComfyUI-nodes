"""Optional diagnostic observation; never changes UltraTex tensor operations.

Disabled for fast production runs. UTX_FULL_DIAG=1 enables all probes.
No per-step CUDA synchronizations or periodic CUDA calls from worker threads.
"""
from __future__ import annotations

import contextvars
import functools
import logging
import os
import statistics
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_CURRENT = contextvars.ContextVar("utx_full_diagnostics", default=None)
_LOG = logging.getLogger("UltraTex")


def active():
    return _CURRENT.get()


def enabled():
    return os.environ.get("UTX_FULL_DIAG", "0").strip().lower() not in ("0", "off", "false", "no")


def _mib(n):
    return round(n / 2**20, 1) if isinstance(n, (int, float)) else None


def _gib(n):
    return round(n / 2**30, 3) if isinstance(n, (int, float)) else None


def _out_dir():
    try:
        import folder_paths
        return Path(folder_paths.get_output_directory()) / "ultratex_diagnostics"
    except Exception:
        return Path.cwd() / "output" / "ultratex_diagnostics"


def safe_model_loaded_size_mib(patcher):
    try:
        fn=getattr(patcher,"loaded_size",None)
        if callable(fn): return _mib(fn())
    except Exception: pass
    return None


def _loaded_models_snapshot():
    """ComfyUI public-ish manager list; read only and optional."""
    result = []
    try:
        import comfy.model_management as mm
        for entry in list(getattr(mm, "current_loaded_models", ())):
            patcher = getattr(entry, "model", None)
            item = {"patcher_type": type(patcher).__name__ if patcher is not None else type(entry).__name__}
            for name in ("loaded_size", "model_size"):
                method = getattr(patcher, name, None)
                if callable(method):
                    try:
                        item[name + "_mib"] = _mib(method())
                    except Exception: pass
            for name in ("device", "load_device", "offload_device", "current_device"):
                v = getattr(patcher, name, None)
                if v is not None: item[name] = str(v)
            result.append(item)
    except Exception as e:
        result = [{"error": str(e)}]
    return result


def _vram_snapshot():
    out = {}
    try:
        import torch
        if torch.cuda.is_available():
            device=torch.cuda.current_device()
            out={"device": int(device),
                 "allocated_mib": _mib(torch.cuda.memory_allocated(device)),
                 "reserved_mib": _mib(torch.cuda.memory_reserved(device)),
                 "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
                 "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved(device))}
            try:
                free,total = torch.cuda.mem_get_info(device)
                out.update({"cuda_driver_free_mib": _mib(free), "cuda_driver_total_mib": _mib(total)})
            except Exception: pass
    except Exception as e:
        out = {"unavailable": str(e)}
    return out


def _ram_snapshot():
    out = {}
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        vm=psutil.virtual_memory()
        mem=proc.memory_info()
        out={"process_rss_mib":_mib(mem.rss),"system_ram_available_mib":_mib(vm.available),
             "system_ram_used_percent":round(vm.percent,1)}
        try:
            i = proc.io_counters()
            out.update({"process_read_bytes":getattr(i,"read_bytes",None),"process_write_bytes":getattr(i,"write_bytes",None)})
        except Exception: pass
    except Exception as e:
        out={"unavailable":str(e)}
    return out


class DiagnosticRun:
    def __init__(self, run_label):
        self.run_label=run_label
        self.start=time.perf_counter()
        self.started_utc=datetime.now(timezone.utc).isoformat()
        self.phase="UTX2.start"
        self.events=[]
        self.samples=[]
        self.steps=[]
        self.errors=[]
        self.notes=[]
        self._pending_events=[]
        self._stop=threading.Event()
        self._thread=None
        self._sample_interval=max(0.3, float(os.environ.get("UTX_DIAG_INTERVAL", "0.5")))
        self.path=None
        self._sampler_error=None
        self._nvml_ok=False
        self._max_samples=12000
        self._output_errors=[]
        self.live_path=_out_dir() / ("utx_DIAGNOSTIC_LIVE_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                                      + "_" + str(os.getpid()) + ".json")
        self._last_checkpoint=0.0

    def t(self):
        return round(time.perf_counter()-self.start,3)

    def mark(self, stage, *, residency=False, **metadata):
        self.phase=stage
        row={"t":self.t(),"event":stage,"cuda":_vram_snapshot(),"ram":_ram_snapshot()}
        if metadata: row["details"]=metadata
        if residency: row["loaded_models"]=_loaded_models_snapshot()
        self.events.append(row)
        _LOG.info("[UTX FULL DIAG] %s | alloc=%sMiB reserved=%sMiB | t=%.1fs",
                  stage,row["cuda"].get("allocated_mib"),row["cuda"].get("reserved_mib"),row["t"])

    def telemetry_loop(self):
        """NVML poll thread: NO torch calls here; no device synchronization."""
        nvml=None; handle=None
        try:
            import pynvml as pynvml_mod
            nvml=pynvml_mod
            nvml.nvmlInit()
            index=0
            try:
                import torch
                # Current device is only read once; sampling thread never calls CUDA functions.
                # Device index 0 is correct on the standard single-GPU ComfyUI setup.
                index=int(os.environ.get("UTX_DIAG_NVML_DEVICE", "0"))
            except Exception: pass
            handle=nvml.nvmlDeviceGetHandleByIndex(index)
            self._nvml_ok=True
        except Exception as err:
            self._sampler_error="NVML unavailable: "+repr(err)
        try:
            import psutil
        except ImportError:
            psutil=None
        while not self._stop.is_set():
            if len(self.samples) >= self._max_samples:
                self._stop.wait(self._sample_interval)
                continue
            row={"t":self.t(),"phase":self.phase}
            if handle is not None:
                for name, fn, converter in (
                    ("gpu_util_percent", lambda:nvml.nvmlDeviceGetUtilizationRates(handle).gpu, float),
                    ("vram_used_mib",lambda:nvml.nvmlDeviceGetMemoryInfo(handle).used, _mib),
                    ("gpu_temp_c",lambda:nvml.nvmlDeviceGetTemperature(handle,nvml.NVML_TEMPERATURE_GPU),float),
                    ("power_w",lambda:nvml.nvmlDeviceGetPowerUsage(handle)/1000.,float),
                    ("pcie_tx_kib_s",lambda:nvml.nvmlDeviceGetPcieThroughput(handle,nvml.NVML_PCIE_UTIL_TX_BYTES),float),
                    ("pcie_rx_kib_s",lambda:nvml.nvmlDeviceGetPcieThroughput(handle,nvml.NVML_PCIE_UTIL_RX_BYTES),float),
                ):
                    try: row[name]=round(converter(fn()),2)
                    except Exception: pass
            if psutil is not None:
                try:
                    row["system_ram_available_mib"]=_mib(psutil.virtual_memory().available)
                    row["system_cpu_percent"]=psutil.cpu_percent(interval=None)
                    row["proc_rss_mib"]=_mib(psutil.Process(os.getpid()).memory_info().rss)
                except Exception: pass
            self.samples.append(row)
            if self.t()-self._last_checkpoint >= 20.0:
                self.checkpoint()
                self._last_checkpoint=self.t()
            self._stop.wait(self._sample_interval)
        if nvml is not None and handle is not None:
            try: nvml.nvmlShutdown()
            except Exception: pass

    def begin(self):
        self.mark("UTX2.start",residency=True,monitor_interval_s=self._sample_interval,
                  backup_live_report=str(self.live_path))
        self.checkpoint()
        self._thread=threading.Thread(target=self.telemetry_loop,daemon=True,name="UTX-VRAM-telemetry")
        self._thread.start()

    def checkpoint(self):
        """Small, robust on-disk heartbeat; survives forced ComfyUI process termination."""
        from .runtime import atomic_json
        try:
            atomic_json(self.live_path, {"status":"running_or_abrupt_exit",
                    "started_utc":self.started_utc,"time_seconds":self.t(),
                    "monitor_note":"Incomplete heartbeat. Upload final utx_full_diagnostic JSON if it exists.",
                    "recent_events":self.events[-20:],"recent_steps":self.steps[-16:],
                    "recent_telemetry":self.samples[-60:],"telemetry_samples_so_far":len(self.samples),
                    "telemetry_error":self._sampler_error})
        except Exception as e:
            self._output_errors.append(str(e))

    def begin_step(self, label, step, dtype, tokens):
        self.phase=f"{label}.diffusion_step_{step}"
        cpu_start=time.perf_counter()
        cuda_start=None
        try:
            import torch
            if torch.cuda.is_available():
                cuda_start=torch.cuda.Event(enable_timing=True)
                cuda_start.record()  # async event; never synchronize inside a step
        except Exception: pass
        return {"cpu_start":cpu_start,"cuda_start":cuda_start,"label":label,"step":step,
                "dtype":str(dtype),"tokens":tokens}

    def end_step(self, ctx):
        end=time.perf_counter()
        row={"t":self.t(),"label":ctx["label"],"step":ctx["step"],
             "cpu_submit_seconds":round(end-ctx["cpu_start"],4),
             "dtype":ctx["dtype"],"tokens":ctx["tokens"],
             "cuda":_vram_snapshot()}
        if ctx["cuda_start"] is not None:
            try:
                import torch
                event=torch.cuda.Event(enable_timing=True)
                event.record()
                self._pending_events.append((row,ctx["cuda_start"],event))
            except Exception: pass
        self.steps.append(row)
        if len(self.steps) <= 120:
            _LOG.info("[UTX FULL DIAG] %s step %d | CPU submission %.2fs | alloc %.0fMiB",
                      ctx["label"],ctx["step"],row["cpu_submit_seconds"],row["cuda"].get("allocated_mib",0))

    def finish(self, error=None):
        if error:
            self.errors.append({"type":type(error).__name__,"message":str(error),
                                "traceback":traceback.format_exception_only(type(error),error)})
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        # We never synchronize to gather incomplete CUDA events when an OOM/interruption occurred.
        for row,a,b in self._pending_events:
            try:
                if b.query(): row["cuda_event_ms"]=round(a.elapsed_time(b),3)
                else: row["cuda_event_pending"]=True
            except Exception: row["cuda_event_unavailable"]=True
        self._pending_events.clear()
        self.mark("UTX2.end",residency=True,success=(error is None))
        self.write()
        if self.path is not None:
            try: self.live_path.unlink(missing_ok=True)
            except Exception: pass

    def write(self):
        from .runtime import atomic_json
        d=_out_dir()
        name="utx_full_diagnostic_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+str(os.getpid())
        p=d/(name+".json")
        vals=[s["gpu_util_percent"] for s in self.samples if isinstance(s.get("gpu_util_percent"),(float,int))]
        vram=[s["vram_used_mib"] for s in self.samples if isinstance(s.get("vram_used_mib"),(float,int))]
        summary={"duration_seconds":self.t(),"telemetry_samples":len(self.samples),
                 "nvml_available":self._nvml_ok,"telemetry_error":self._sampler_error,
                 "gpu_util_avg_percent":round(statistics.mean(vals),1) if vals else None,
                 "gpu_util_min_percent":min(vals) if vals else None,
                 "gpu_util_low_below_30_percent_samples":sum(v<30 for v in vals) if vals else None,
                 "nvml_vram_max_mib":max(vram) if vram else None,
                 "steps":len(self.steps)}
        payload={"format":"UltraTex full run diagnostic v1","started_utc":self.started_utc,
                 "runtime":{"python":__import__("sys").version,"torch":self._torch_version(),
                            "tiled_lora_env":os.environ.get("UTX_TILED_LORA","1"),
                            "note":"CUDA event time is GPU time for the default recorded stream; async offload streams can differ. CPU submit time is not synchronized GPU time. NVML under WDDM may not include shared VRAM. PyTorch allocated/reserved does not measure Windows shared memory."},
                 "summary":summary,"events":self.events,"steps":self.steps,
                 "telemetry":self.samples,"errors":self.errors,"notes":self.notes,
                 "checkpoint_errors":self._output_errors}
        try:
            atomic_json(p,payload)
            self.path=str(p)
            _LOG.warning("[UTX FULL DIAG] REPORT: %s",p)
            print("[UTX FULL DIAG] REPORT:",p,flush=True)
        except Exception as e:
            _LOG.error("[UTX FULL DIAG] Unable to save report: %s",e)

    @staticmethod
    def _torch_version():
        try:
            import torch
            return str(torch.__version__)
        except Exception: return "unavailable"


def monitor_stage(label, fn, *args, **kwargs):
    """Time a single existing call; propagates results/errors unchanged."""
    d=active()
    if d is None: return fn(*args,**kwargs)
    d.mark(label+".begin")
    start=time.perf_counter()
    try:
        value=fn(*args,**kwargs)
    except BaseException as exc:
        d.mark(label+".error",seconds=round(time.perf_counter()-start,3),error=str(exc))
        raise
    d.mark(label+".end",seconds=round(time.perf_counter()-start,3))
    return value


def diagnose_node(fn):
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        if not enabled(): return fn(*args,**kwargs)
        if active() is not None: return fn(*args,**kwargs)
        d=DiagnosticRun(getattr(fn,"__qualname__",str(fn)))
        token=_CURRENT.set(d)
        d.begin()
        err=None
        try:
            return fn(*args,**kwargs)
        except BaseException as ex:
            err=ex
            raise
        finally:
            try: d.finish(err)
            except Exception: _LOG.exception("[UTX FULL DIAG] Unexpected report failure")
            _CURRENT.reset(token)
    return wrapped
