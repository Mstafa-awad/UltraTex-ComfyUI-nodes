"""Sampling only. No VAE, rasterizer, text encoder, or bake calls in this module."""
from __future__ import annotations

import gc
import logging
import math
import threading
import time
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import torch

from .contracts import Samples, SampleResult
from .models import validate_backbone
from .runtime import Store, atomic_json, fingerprint, interrupt, tensor_from
from .diagnostics import active as _active_diag, safe_model_loaded_size_mib as _safe_model_size_mib

_SAMPLING_LOCK=threading.RLock()
_LOG=logging.getLogger("UltraTex")


def schedule(steps,resolution,denoise=1.,has_init=False):
    if steps<1 or not 0<=denoise<=1:
        raise ValueError("Steps must be positive and denoise between 0 and 1")
    if denoise<1 and not has_init:
        raise ValueError("Denoise below 1 requires encoded init views")
    if denoise==0:
        return [0.]
    # Training-era UNO schedule parameter, intentionally NOT the DiT token count.
    length=((resolution*3)//8)*((resolution*2)//8)//256
    mu=(1.15-.5)/(4096-256)*(length-256)+.5
    raw=torch.linspace(1,0,steps+1)
    values=(math.exp(mu)/(math.exp(mu)+(1/raw-1))).tolist()
    if denoise<1:
        # An explicit starting point handles denoise values smaller than the last scheduled step.
        values=[denoise]+[t for t in values if 0<=t<denoise]
    return values


@lru_cache(maxsize=8)
def sparse_probe(device_string,dtype_string,torch_version):
    from ..vendor import optim
    device=torch.device(device_string)
    dtype=getattr(torch,dtype_string)
    try:
        with torch.inference_mode():
            q=torch.zeros((1,2,2048,128),device=device,dtype=dtype)
            out=optim._sla(.2)(q,q,q)
            if not torch.isfinite(out).all():
                raise RuntimeError("Sparse probe returned nonfinite values")
            if device.type=="cuda":
                torch.cuda.synchronize(device)
        return None
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def model_info(model):
    try:
        info=model.get_attachment("utx_model_info")
    except (AttributeError,KeyError):
        info=None
    if info is None:
        raise ValueError("Connect MODEL through UTX Apply Texture LoRA before sampling")
    return info


def _dtype(model):
    dtype=getattr(model.model,"manual_cast_dtype",None) or model.model.get_dtype()
    return dtype if dtype in (torch.float16,torch.bfloat16,torch.float32) else torch.bfloat16


def _streamed_gguf_active(model, chunk):
    """True only if ALL single blocks can use the memory-saving path."""
    from ..vendor.optim import _streaming_enabled, _gguf_linear_can_split
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = getattr(dm, "single_blocks", ())
    return bool(chunk and _streaming_enabled() and blocks
                and all(_gguf_linear_can_split(getattr(b, "linear1", None)) for b in blocks))


def _effective_activation_factor(model, policy, chunk):
    base = policy.activation_factor if chunk else max(28., policy.activation_factor)
    return min(base, max(6., base-4.)) if _streamed_gguf_active(model, chunk) else base


def _estimate(job,model,policy,chunk):
    n=job.pack.position_ids.shape[1]+job.positive.shape[1]
    dtype=_dtype(model)
    # The observed first-pass peak is dominated by full-sequence MLP storage.
    # Removing a ~3GiB buffer permits a conservative ~2GiB reservation reduction
    # at 65K tokens, leaving one GiB of safety for other temporary tensors.
    factor=_effective_activation_factor(model,policy,chunk)
    activations=int(n*4096*torch.empty((),dtype=dtype).element_size()*factor)
    block_map=int(32*math.ceil(n/128)*math.ceil(n/64)*12)
    return activations+block_map+policy.gpu_headroom_mb*2**20


def _validate_text(job):
    # Validate before loading diffusion weights or compiling a GPU kernel.
    if job.positive.shape != (1,512,12288):
        raise ValueError(f"FLUX.2 Klein base 9B needs Qwen3-8B conditioning "
                         f"(1, 512, 12288), got {job.positive.shape}. "
                         "Use the 9B-compatible Qwen3-8B encoder, not Qwen3-4B.")
    if job.negative is not None and job.negative.shape != (1,512,12288):
        raise ValueError(f"Negative conditioning has incompatible shape {job.negative.shape}; "
                         "Klein base 9B needs (1, 512, 12288).")


def _sample_job(model,job,steps,guidance,denoise,backend,topk,chunk,policy,controller=None):
    import comfy.model_management as mm
    import comfy.utils
    from ..vendor import optim
    pack=job.pack
    device=mm.get_torch_device()
    dtype=_dtype(model)
    times=schedule(steps,pack.conditions.gbuffer.plan.sample_size,denoise,pack.init_tokens is not None)
    _validate_text(job)
    positive=tensor_from(job.positive).to(device=device,dtype=dtype)
    negative=tensor_from(job.negative).to(device=device,dtype=dtype) if job.negative else None
    use_cfg=negative is not None and guidance!=1. and not torch.equal(positive,negative)
    ids=tensor_from(pack.position_ids).to(device)
    refs=tensor_from(pack.references).to(device=device,dtype=dtype)
    fg=tensor_from(pack.foreground_indices).long()
    generator=torch.Generator("cpu").manual_seed(job.seed)
    # Draw the full canonical sequence on CPU before selecting, preserving seed/layout semantics.
    noise=torch.randn((1,pack.grid_h*pack.grid_w,128),generator=generator,dtype=torch.float32)[:,fg].to(device)
    initial=tensor_from(pack.init_tokens).to(device) if pack.init_tokens else None
    keep=tensor_from(pack.keep_weights).to(device) if pack.keep_weights else None
    x=noise.clone() if initial is None else (1-times[0])*initial+times[0]*noise
    text_ids=torch.zeros((1,positive.shape[1],4),device=device)
    text_ids[...,3]=torch.arange(positive.shape[1],device=device)
    options=dict(model.model_options.get("transformer_options",{}))
    counts={"sparse_calls":0,"dense_calls":0}
    if backend=="sparse":
        native_override=optim.sparse_attention_override(topk)
        def attention(func,q,k,v,heads,**kwargs):
            usable=(kwargs.get("skip_reshape",False) and kwargs.get("mask") is None and q.ndim==4 and
                    q.shape[-1]==128 and q.shape[2]==k.shape[2]>=1024 and q.is_cuda)
            if not usable:
                raise RuntimeError("Sparse backend cannot handle this attention layout; select dense_debug explicitly for diagnostics")
            counts["sparse_calls"]+=1
            return native_override(func,q,k,v,heads,**kwargs)
        options["optimized_attention_override"]=attention
    else:
        def attention(func,*args,**kwargs):
            counts["dense_calls"]+=1
            return func(*args,**kwargs)
        options["optimized_attention_override"]=attention
    dm=model.model.diffusion_model
    pbar=comfy.utils.ProgressBar(max(len(times)-1,1))
    diag=_active_diag()
    sample_label=diag.phase.rsplit(".sampling",1)[0] if diag is not None else "UTX2.sample"
    if diag:
        diag.mark(sample_label+".prepared", total_token_sequence=int(ids.shape[1]+positive.shape[1]),
                  normal_tokens=int(pack.normal_count),reference_tokens=int(pack.reference_count),
                  selected_foreground_tokens=int(pack.target_count),text_tokens=int(positive.shape[1]),
                  grid_height=int(pack.grid_h),grid_width=int(pack.grid_w),
                  full_grid_tokens=int(pack.grid_h*pack.grid_w),
                  foreground_ratio=round(float(pack.target_count)/max(1,pack.grid_h*pack.grid_w),4),
                  cfg_sequential=bool(use_cfg), seed=job.seed, steps=int(len(times)-1),
                  token_chunk=chunk, dtype=str(dtype), sparse_retention=float(topk))
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started=time.perf_counter()
    # Experimental, reversible optimization: standard GGUF LoRA applies FP32 row tiles.
    # Disable with UTX_TILED_LORA=0; no changes to framing, tokens or sampler math.
    from ..vendor.gguf_tiled_lora import patch_gguf_lora
    with patch_gguf_lora(dm) as tiled_report, optim.chunked_model(dm,options,chunk) as transformer_options:
        for step_index, (t,next_t) in enumerate(zip(times[:-1],times[1:]),1):
            interrupt()
            step_started=time.perf_counter() if optim.PROF.enabled else None
            diag_step=diag.begin_step(sample_label,step_index,dtype,int(ids.shape[1]+positive.shape[1])) if diag else None
            tv=torch.full((1,),t,device=device,dtype=dtype)
            # Rebuild input for each call: upstream blocks may update their input buffer in place.
            def predict(ctx):
                inp=torch.cat((x.to(dtype),refs),dim=1)
                return dm.forward_orig(inp,ids,ctx,text_ids,tv,None,guidance=None,
                                       transformer_options=transformer_options)[:,:pack.target_count].float()
            pred=predict(positive)
            if use_cfg:
                uncond=predict(negative)
                pred=uncond+guidance*(pred-uncond)
                del uncond
            x=x+(next_t-t)*pred
            del pred
            if keep is not None:
                known=(1-next_t)*initial+next_t*noise
                x=keep*known+(1-keep)*x
            if controller is not None:controller.after_step(model)
            pbar.update(1)
            if diag_step is not None:
                diag.end_step(diag_step)
            if step_started is not None:
                optim.PROF.report(step_index,time.perf_counter()-step_started)
    if diag:
        diag.mark(sample_label+".diffusion_complete", tiled_lora_casts=int(tiled_report["tiled_casts"]),
                  tiled_lora_fallbacks=int(tiled_report["fallback_casts"]))
    if tiled_report["tiled_casts"] or tiled_report["fallback_casts"]:
        _LOG.info("[UTX tiled LoRA] tiled GGUF casts=%d; unsupported fallback casts=%d; "
                  "set UTX_TILED_LORA=0 to compare original path",
                  tiled_report["tiled_casts"], tiled_report["fallback_casts"])
    if device.type=="cuda":
        torch.cuda.synchronize(device)
    result=x.detach().cpu().numpy()
    elapsed=time.perf_counter()-started
    report=dict(stage="sample",job_id=job.identity,seed=job.seed,task=job.task,steps_requested=steps,
                steps_executed=len(times)-1,denoise=denoise,guidance=guidance,cfg="sequential" if use_cfg else "disabled",
                backend=backend,dtype=str(dtype),topk=topk,token_chunk=chunk,
                target_tokens=pack.target_count,normal_tokens=pack.normal_count,reference_tokens=pack.reference_count,
                text_tokens=positive.shape[1],sample_size=pack.conditions.gbuffer.plan.sample_size,
                denoise_seconds=elapsed,seconds_per_step=elapsed/max(len(times)-1,1),
                tiled_lora_active=bool(__import__("os").environ.get("UTX_TILED_LORA", "1").lower() not in ("0","off","no","false")),
                tiled_lora_casts=tiled_report["tiled_casts"],
                tiled_lora_fallback_casts=tiled_report["fallback_casts"],**counts)
    if device.type=="cuda":
        peak_allocated=torch.cuda.max_memory_allocated(device)
        peak_reserved=torch.cuda.max_memory_reserved(device)
        total_physical=torch.cuda.get_device_properties(device).total_memory
        report.update(peak_gpu_allocated=peak_allocated,
                      peak_gpu_reserved=peak_reserved,
                      physical_vram_bytes=total_physical,
                      peak_reserved_exceeds_physical_vram=peak_reserved>total_physical,
                      gpu_accounting_note="PyTorch allocator statistics are not a measured PCIe transfer count")
        if peak_reserved>total_physical:
            _LOG.warning("[UTX memory] PyTorch reserved %.2f GiB vs %.2f GiB physical VRAM. "
                         "Windows may be paging into shared GPU memory; successful 2K generation "
                         "does not guarantee fast dedicated-VRAM execution.",
                         peak_reserved/2**30,total_physical/2**30)
    return result,report


@torch.inference_mode()
def sample_workset(model,workset,policy,steps=25,guidance=4.,denoise=1.,backend="sparse",topk=.2,material_model=None):
    import comfy.model_management as mm
    bindings={model_info(model).task:model}
    if material_model is not None:
        if model_info(material_model).task in bindings:
            raise ValueError("The two MODEL branches must select different albedo/MR tasks")
        bindings[model_info(material_model).task]=material_model
        if material_model.model is not model.model:
            raise ValueError("Albedo and MR models must be clones from the same backbone loader, sharing one base")
    for binding in bindings.values():
        validate_backbone(binding)
    if not workset.jobs or any(job.task not in bindings for job in workset.jobs):
        raise ValueError("Workset contains tasks without a matching albedo/MR MODEL binding")
    if backend not in ("sparse","dense_debug") or not 0<topk<=1:
        raise ValueError("Invalid attention backend or sparse retention")
    for job in workset.jobs:
        _validate_text(job)
    device=mm.get_torch_device()
    if backend=="sparse":
        error=sparse_probe(str(device),str(_dtype(model)).split(".")[-1],torch.__version__)
        if error:
            raise RuntimeError("UltraTex sparse backend unavailable: "+error+
                               ". Install a matching triton-windows in ComfyUI's Python; dense_debug is an explicit alternative.")
    store=Store(policy)
    run_id=fingerprint("run-v4-independent-qkv",[j.identity for j in workset.jobs],
                       {task:{"info":asdict(model_info(binding)),"dtype":str(_dtype(binding)),
                              "patches_uuid":str(getattr(binding,"patches_uuid","")),
                              "options":repr(getattr(binding,"model_options",{}))}
                        for task,binding in bindings.items()},
                       steps,guidance,denoise,backend,topk,policy.token_chunk,
                       "streamed-gguf-mlp-v1" if __import__("os").environ.get("UTX_STREAMED_MLP", "1").lower() not in ("0", "false", "off", "no") else "legacy-gguf-mlp")
    # Diagnostics must perform REAL denoising, even if the same seed/settings
    # previously produced a persistent disk-cache hit. This changes only the
    # temporary cache key; sampling tokens, seed, weights and mathematics are untouched.
    diag_run=_active_diag()
    bench_fresh=__import__("os").environ.get("UTX_BENCH_FRESH","0").strip().lower() in ("1","on","true","yes")
    if bench_fresh or (diag_run and __import__("os").environ.get("UTX_DIAG_FORCE_SAMPLE","1").strip().lower() not in ("0","off","false","no")):
        run_id=fingerprint(run_id,"utx-diagnostic-fresh",time.time_ns())
        if diag_run:
            diag_run.mark("UTX2.sampling.force_fresh_cache",cache_reuse_disabled=True)
    manifest=store.root/("workset-"+run_id+".json")
    done={}
    completed=[]
    load_requests=0
    start=time.perf_counter()
    def progress(status,error=None):
        atomic_json(manifest,dict(version=1,status=status,completed=completed,total_jobs=len(workset.jobs),
                                 model_load_requests=load_requests,error=error))
    progress("running")
    with _SAMPLING_LOCK:
        try:
            for task,binding in bindings.items():
                group=[(i,j) for i,j in enumerate(workset.jobs) if j.task==task]
                if not group:
                    continue
                pending=[job for _,job in group if store.lookup(fingerprint(run_id,job.identity)) is None]
                # One manager request for the largest activation envelope in the task group.
                diag=_active_diag()
                if diag:
                    for _,j in group:
                        token_total=int(j.pack.position_ids.shape[1]+j.positive.shape[1])
                        diag.mark("UTX2.sampling.job_tokens", task=task, job_id=j.identity,
                                  selected_fg_tokens=int(j.pack.target_count),normal_tokens=int(j.pack.normal_count),
                                  reference_tokens=int(j.pack.reference_count),text_tokens=int(j.positive.shape[1]),
                                  total_transformer_tokens=token_total,
                                  grid_tokens=int(j.pack.grid_h*j.pack.grid_w),
                                  fg_fraction=round(j.pack.target_count/max(j.pack.grid_h*j.pack.grid_w,1),4),
                                  target_resolution=int(j.pack.conditions.gbuffer.plan.sample_size))
                memory=max((_estimate(job,binding,policy,policy.token_chunk) for job in pending),default=0)
                from .adaptive import Controller
                controller=Controller(policy,device)
                memory=controller.reservation(memory) if pending else memory
                if device.type=="cuda" and memory>torch.cuda.get_device_properties(device).total_memory:
                    raise MemoryError(f"Estimated activation/headroom envelope {memory/2**30:.2f} GiB exceeds GPU capacity. "
                                      "Reduce foreground occupancy or resolution, or calibrate activation_factor on measured runs.")
                if diag:
                    diag.mark("UTX2.sampling.reservation",activation_and_headroom_bytes=int(memory),
                              activation_and_headroom_gib=round(memory/2**30,3),
                              headroom_mb=policy.gpu_headroom_mb, activation_factor=policy.activation_factor,
                              effective_activation_factor=_effective_activation_factor(binding,policy,policy.token_chunk),
                              streamed_gguf_mlp=_streamed_gguf_active(binding,policy.token_chunk),
                              token_chunk=policy.token_chunk,pending_jobs=len(pending),
                              loaded_size_mib=_safe_model_size_mib(binding),
                              residency=True)
                if pending:
                    if steps<10 and denoise==1:
                        _LOG.warning("UltraTex: %d steps is a preview for the undistilled Klein base 9B. "
                                     "Start with 25 steps for texture quality.",steps)
                    load_started=time.perf_counter()
                    if diag:
                        diag.mark("UTX2.sampling.model_load.begin",residency=True,manager_memory_required_bytes=int(memory))
                    try:
                        from .gguf_load import storage_context,research_enabled
                        with storage_context(binding,research_enabled(policy)) as detach_metrics:
                            mm.load_models_gpu([binding],memory_required=memory)
                    finally:
                        if diag: diag.mark("UTX2.sampling.model_load.end",residency=True,
                                           load_seconds=round(time.perf_counter()-load_started,3))
                    load_seconds=time.perf_counter()-load_started
                    load_requests+=1
                for index,job in group:
                    interrupt()
                    key=fingerprint(run_id,job.identity)
                    cached=store.lookup(key)
                    if cached is not None:
                        report={"stage":"sample","job_id":job.identity,"seed":job.seed,"cache_hit":True}
                        latent=cached
                    else:
                        chunk=policy.token_chunk
                        try:
                            with controller.optimizations(binding,job.pack.position_ids.shape[1]+job.positive.shape[1],_dtype(binding)):
                                values,report=_sample_job(binding,job,steps,guidance,denoise,backend,topk,chunk,policy,controller)
                        except torch.cuda.OutOfMemoryError as error:
                            if diag:
                                diag.mark("UTX2.sampling.CUDA_OOM",residency=True,error=str(error))
                            # Recovery occurs only on failure. Clear traceback references before retrying.
                            error.__traceback__=None
                            gc.collect()
                            mm.soft_empty_cache()
                            chunk=max(256,chunk//2) if chunk else 2048
                            mm.load_models_gpu([binding],memory_required=memory)
                            load_requests+=1
                            with controller.optimizations(binding,job.pack.position_ids.shape[1]+job.positive.shape[1],_dtype(binding)):
                                values,report=_sample_job(binding,job,steps,guidance,denoise,backend,topk,chunk,policy,controller)
                            report["oom_retry"]=True
                        report.update(activation_and_headroom_bytes=memory,
                                      adaptive_decisions=controller.records,
                                      model_group_load_seconds=load_seconds,
                                      gpu_headroom_mb=policy.gpu_headroom_mb,
                                      activation_factor=policy.activation_factor)
                        if callable(getattr(binding,"loaded_size",None)):
                            report["loaded_weight_bytes"]=binding.loaded_size()
                        latent=store.put(values,key)
                        del values
                    if diag:
                        diag.mark("UTX2.sampling.job_complete",residency=True,job_id=job.identity,
                                  cache_hit=bool(cached is not None),
                                  sample_report={k:v for k,v in report.items()
                                    if k in ("denoise_seconds","seconds_per_step","peak_gpu_allocated",
                                             "peak_gpu_reserved","target_tokens","normal_tokens","reference_tokens",
                                             "tiled_lora_casts","tiled_lora_fallback_casts","load_seconds",
                                             "activation_and_headroom_bytes","model_group_load_seconds")})
                    done[index]=SampleResult(job,latent,report)
                    completed.append(dict(index=index,job_id=job.identity,seed=job.seed,
                                          latent=asdict(latent),report=report))
                    progress("running")
            progress("complete")
        except BaseException as error:
            progress("interrupted_or_failed",str(error))
            raise
    results=tuple(done[i] for i in range(len(workset.jobs)))
    return Samples(results,str(manifest)),dict(stage="sample_workset",jobs=len(results),wall_seconds=time.perf_counter()-start,
                model_load_requests=load_requests,manifest=str(manifest),
                note="Manager requests are instrumented; internal layer-transfer counts require ComfyUI profiler logs.")
