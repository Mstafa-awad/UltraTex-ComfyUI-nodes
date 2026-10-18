"""Execution-scoped memory decisions; ComfyUI remains the sole weight manager."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from contextlib import contextmanager
import os
import logging
import torch

MiB=2**20
LOG=logging.getLogger('UltraTex')

@dataclass(frozen=True)
class Snapshot:
    total:int
    free:int
    allocated:int
    reserved:int
    system_available:int
    capability:tuple[int,int]=(0,0)
    dynamic:bool=False
    local_budget:int=0
    local_usage:int=0
    shared_usage:int=0

def snapshot(device):
    import psutil
    if device.type!='cuda' or torch.version.cuda is None:return None
    from .gpu_budget import windows_budget
    free,total=torch.cuda.mem_get_info(device)
    w=windows_budget(torch.cuda.get_device_name(device)) or {}
    import comfy.memory_management as memory
    return Snapshot(total,free,torch.cuda.memory_allocated(device),torch.cuda.memory_reserved(device),
                    psutil.virtual_memory().available,torch.cuda.get_device_capability(device),
                    bool(getattr(memory,'aimdo_enabled',False)),w.get('local_budget',0),w.get('local_usage',0),w.get('shared_usage',0))

@dataclass(frozen=True)
class Decision:
    mode:str
    capacity:int
    headroom:int
    spendable:int
    vae_batch:int
    reason:str

def decide(state,policy):
    mode=getattr(policy,'performance_mode','adaptive')
    if state is None or mode=='original':return Decision('original',0,0,0,1,'native fallback')
    limit=getattr(policy,'max_vram_mb',0)*MiB
    capacity=min(state.total,limit or state.total,state.local_budget or state.total)
    headroom=max(policy.gpu_headroom_mb*MiB,int(capacity*.08))
    if mode=='conservative':headroom=max(headroom,1024*MiB,int(capacity*.12))
    reusable=max(0,state.reserved-state.allocated)
    free=min(state.free+reusable,max(0,capacity-state.allocated))
    if state.local_budget:
        free=min(free,max(0,state.local_budget-state.local_usage)+reusable)
    spendable=max(0,free-headroom)
    pressure=state.shared_usage>64*MiB or state.system_available<policy.system_headroom_mb*MiB
    return Decision(mode,capacity,headroom,spendable,1,'pressure fallback' if pressure else 'bounded available-memory plan')

class Controller:
    """Owns only scalar records for one top-level operation, never model tensors."""
    def __init__(self,policy,device):self.policy=policy;self.device=device;self.records=[]
    def plan(self,stage):
        state=snapshot(self.device)
        decision=decide(state,self.policy)
        self.records.append(dict(stage=stage,snapshot=asdict(state) if state else None,decision=asdict(decision)))
        if getattr(self.policy,'adaptive_diagnostics',False):LOG.info('UltraTex %s: capacity %.0f MiB, headroom %.0f MiB, spendable %.0f MiB',stage,decision.capacity/MiB,decision.headroom/MiB,decision.spendable/MiB)
        return decision.vae_batch
    def reservation(self,estimate):
        state=snapshot(self.device)
        if state is None or getattr(self.policy,'performance_mode','adaptive')=='original':return estimate
        decision=decide(state,self.policy)
        if estimate+128*MiB>decision.capacity:
            raise MemoryError(f'UltraTex needs an estimated {estimate/MiB:.0f} MiB activation/headroom reservation plus weight workspace; current GPU budget is {decision.capacity/MiB:.0f} MiB. Quality settings were preserved. Use original fallback only if you accept its native memory behavior.')
        # CUDA free memory / ComfyUI already account for driver pressure.
        # Add only the user-requested exclusion; adding physical-DXGI budget
        # again double-counts WDDM pressure and needlessly evicts packed weights.
        user_cap=getattr(self.policy,'max_vram_mb',0)*MiB
        exclusion=max(0,(state.local_budget or state.total)-user_cap) if user_cap else 0
        reservation=estimate+exclusion+max(0,decision.headroom-self.policy.gpu_headroom_mb*MiB)
        self.records.append(dict(stage='flux_reservation',snapshot=asdict(state),manager_reservation=reservation))
        self.reserved_envelope=reservation
        self.activation_estimate=estimate
        self.initial_shared=state.shared_usage
        if getattr(self.policy,'adaptive_diagnostics',False):LOG.info('UltraTex FLUX: manager reservation %.0f MiB, budget %.0f MiB',reservation/MiB,decision.capacity/MiB)
        return reservation

    def needs_bounded_encode(self,size,element_bytes=2):
        """Use bounded convolution only when the native measured envelope will not fit.

        On roomy budgets retain the native reservation: spending encoder savings
        on additional packed weights raised later peaks without useful speed.
        """
        state=snapshot(self.device)
        if state is None or getattr(self.policy,'performance_mode','adaptive')=='original':return False
        decision=decide(state,self.policy)
        native=int(5680*MiB*(size/2048)**2*(element_bytes/2))+decision.headroom+256*MiB
        return native>decision.capacity
    def vae_encode_guard(self,size,element_bytes=2,bounded=False):
        state=snapshot(self.device)
        if state is None or getattr(self.policy,'performance_mode','adaptive')=='original':return
        decision=decide(state,self.policy)
        # Conservative measured envelope for the exact dense 2048 Flux2 BF16
        # encoder on the retained runtime. Never switch to a different tile grid.
        envelope=4096 if bounded else 5680
        required=int(envelope*MiB*(size/2048)**2*(element_bytes/2))+decision.headroom+256*MiB
        self.records.append(dict(stage='vae_encode_preflight',required=required,capacity=decision.capacity))
        if required>decision.capacity:
            raise MemoryError(f'UltraTex unchanged dense VAE encode has a conservative {required/MiB:.0f} MiB envelope at {size}px; current GPU budget is {decision.capacity/MiB:.0f} MiB. This exact-quality path is not validated within that budget; resolution and tiling were not changed.')
    def after_step(self,model):
        if getattr(self.policy,'performance_mode','adaptive')=='original':return
        state=snapshot(self.device)
        if state is None:return
        decision=decide(state,self.policy)
        envelope=getattr(self,'activation_estimate',self.reserved_envelope)
        if envelope+128*MiB>decision.capacity:
            raise MemoryError('UltraTex GPU budget fell below the unchanged-quality activation envelope. Retry after GPU memory pressure recovers; no quality settings were reduced.')
        reusable=max(0,state.reserved-state.allocated)
        active_local=max(0,state.local_usage-reusable)
        over=max(0,active_local-(decision.capacity-decision.headroom//2)) if state.local_budget else 0
        shared_growth=max(0,state.shared_usage-getattr(self,'initial_shared',0))
        if shared_growth>128*MiB:over=max(over,shared_growth)
        if over>128*MiB and getattr(self,'demotions',0)<2:
            import comfy.model_management as mm
            self.reserved_envelope+=over
            mm.load_models_gpu([model],memory_required=self.reserved_envelope)
            self.demotions=getattr(self,'demotions',0)+1
            self.records.append(dict(stage='flux_pressure_demote',snapshot=asdict(state),manager_reservation=self.reserved_envelope))
    @contextmanager
    def optimizations(self,model,n,dtype):
        state=snapshot(self.device);mode=getattr(self.policy,'performance_mode','adaptive')
        packed=model.model_size() if callable(getattr(model,'model_size',None)) else 0
        decision=decide(state,self.policy)
        constrained=bool(state and (packed+n*4096*2*8>decision.capacity-decision.headroom or n*4096*2*8>decision.spendable))
        from ..vendor.optim import supports_bounded_qk_norm
        dm=getattr(getattr(model,'model',None),'diffusion_model',None)
        paired=bool(mode!='original' and constrained and state.capability>=(8,0) and dtype==torch.bfloat16
                    and supports_bounded_qk_norm(dm) and verify_rope(self.device))
        flags=dict(UTX_FUSED_ROPE='1' if paired else '0',UTX_QK_NORM_CHUNK='8192' if paired else '0',
                   UTX_MLP_OUTPUT_REUSE='1' if paired else '0',UTX_Q4_FUSED='0',UTX_RESIDENCY_CREDIT_MB='0')
        previous={key:os.environ.get(key) for key in flags}
        os.environ.update(flags)
        self.records.append(dict(stage='flux_strategy',mode=mode,constrained=constrained,paired=paired,mlp_chunk=self.policy.token_chunk,packed_bytes=packed))
        try:yield
        finally:
            for key,value in previous.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value

_VERIFIED={}
def verify_rope(device):
    key=(torch.cuda.get_device_name(device),torch.cuda.get_device_capability(device),torch.__version__)
    if key in _VERIFIED:return _VERIFIED[key]
    try:
        from comfy.ldm.flux.math import apply_rope as native
        from ..vendor.rope_exact import apply_rope
        with torch.inference_mode():
            generator=torch.Generator(device=device).manual_seed(317)
            q=torch.randn(1,1,16384,128,device=device,dtype=torch.bfloat16,generator=generator)
            k=torch.randn(q.shape,device=device,dtype=q.dtype,generator=generator)
            theta=torch.randn(1,1,16384,64,device=device,generator=generator)
            cs,sn=theta.cos(),theta.sin()
            pe=torch.stack((cs,-sn,sn,cs),-1).reshape(1,1,16384,64,2,2)
            expected=native(q,k,pe);actual=apply_rope(q,k,pe)
            valid=actual is not None and all(torch.equal(a,b) for a,b in zip(expected,actual))
    except Exception as error:
        LOG.info('UltraTex native rotary fallback: %s',error);valid=False
    _VERIFIED[key]=valid
    return valid
