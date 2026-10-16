"""Opt-in full-context VAE TEST; original tiled strategy remains the default.
Full-frame context differs numerically from the three-pass tiled decoder.
"""
from contextlib import contextmanager
import os
import torch
from .vae_conv import bounded_convolutions
from .vae_consuming import consume_residuals

def requested(policy):
 return os.environ.get('UTX_FULL_CONTEXT_VAE','0')=='1' and policy.performance_mode!='original'

def eligibility(vae,policy):
 if not requested(policy):return False,'disabled'
 if type(vae.first_stage_model).__name__!='AutoencoderKL' or getattr(vae,'vae_dtype',None)!=torch.bfloat16:return False,'unsupported VAE'
 if str(vae.output_device)!='cpu' or getattr(vae,'handles_tiling',True):return False,'unsupported VAE interface'
 if not getattr(vae,'utx_decoder_hash',None):return False,'foreground decoder required'
 if not torch.cuda.is_available() or torch.device(vae.device).type!='cuda':return False,'CUDA required'
 if torch.cuda.get_allocator_backend()!='cudaMallocAsync':return False,'untested allocator'
 fraction=torch.cuda.get_per_process_memory_fraction(vae.device)
 if not 0.<fraction<=1.:return False,'explicit startup allocator cap required; see TEST instructions'
 total=torch.cuda.get_device_properties(vae.device).total_memory
 required=(6400+max(768,policy.gpu_headroom_mb))*2**20
 if total<required:return False,'physical memory below tested full-frame envelope'
 if policy.max_vram_mb and policy.max_vram_mb<6144:return False,'policy budget below tested full-frame envelope'
 return True,'explicit TEST mode'

@contextmanager
def full_context_session(vae,policy,*,workload_supported=True):
 enabled,reason=eligibility(vae,policy) if workload_supported else (False,'unsupported workload; original decoder retained')
 state=dict(enabled=enabled,reason=reason,disabled=False,last_used_full=False,full_frames=0,fallbacks=0,rows=128,context_changed=enabled)
 if not enabled:
  yield state;return
 names=('memory_used_decode','decode_tiled_')
 saved=[(n,n in vae.__dict__,vae.__dict__.get(n)) for n in names]
 native_estimate=vae.memory_used_decode;native_tiled=vae.decode_tiled_
 def estimate(shape,dtype):
  if len(shape)==4 and shape[0]==1 and tuple(shape[-2:])==(128,128):return 6400*2**20
  return native_estimate(shape,dtype)
 def tiled_fallback(*a,**kw):
  state.update(disabled=True,last_used_full=False);state['fallbacks']+=1
  return native_tiled(*a,**kw)
 vae.memory_used_decode=estimate;vae.decode_tiled_=tiled_fallback
 try:
  with bounded_convolutions(vae.first_stage_model.decoder,rows=128) as convolutions,consume_residuals(vae.first_stage_model.decoder) as residuals:
   state.update(convolutions=convolutions,residuals=residuals)
   # Respect tighter caller caps; restore the process's existing native setting.
   previous_fraction=torch.cuda.get_per_process_memory_fraction(vae.device)
   total=torch.cuda.get_device_properties(vae.device).total_memory
   fraction=min(previous_fraction,5632*2**20/total)
   state.update(allocator_cap_mib=fraction*total/2**20,previous_allocator_fraction=previous_fraction)
   import comfy.model_management as mm
   mm.soft_empty_cache(force=True)
   torch.cuda.set_per_process_memory_fraction(fraction,vae.device)
   hooks=[]
   state['pool_trims']=0
   def trim_before(module,args):
    mm.soft_empty_cache(force=True);state['pool_trims']+=1
   for module in vae.first_stage_model.decoder.modules():
    if type(module).__name__ in ('ResnetBlock','Upsample'):
     hooks.append(module.register_forward_pre_hook(trim_before))
   try:yield state
   finally:
    for hook in hooks:hook.remove()
    torch.cuda.set_per_process_memory_fraction(previous_fraction,vae.device)
 finally:
  for n,had,value in saved:
   if had:setattr(vae,n,value)
   else:delattr(vae,n)

def decode_full(vae,tile,state):
 import comfy.model_management as mm
 mm.soft_empty_cache(force=True)
 state['last_used_full']=True
 value=vae.decode(tile)
 if state['last_used_full']:state['full_frames']+=1
 return value
