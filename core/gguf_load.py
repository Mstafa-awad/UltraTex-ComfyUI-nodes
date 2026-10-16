"""Research: detach offloaded GGUF storage without a GPU round trip."""
from contextlib import contextmanager
import os
import torch

def research_enabled(policy=None):
    return os.environ.get('UTX_ARCH_RESEARCH','0')=='1' and (policy is None or policy.performance_mode!='original')

@contextmanager
def storage_context(patcher,enabled=True,role='transformer'):
    if not enabled:
        yield {'strategy':'native','reason':'research disabled'};return
    mode=os.environ.get('UTX_ARCH_GGUF_STORAGE','cpu_copy')
    if mode not in ('native','cpu_copy','mmap','hybrid'):
        raise ValueError('UTX_ARCH_GGUF_STORAGE must be native, cpu_copy, mmap or hybrid')
    if mode=='hybrid':mode='mmap' if role=='text' else 'cpu_copy'
    context=(retain_mmap(patcher) if enabled and mode=='mmap' else
             detach_on_cpu(patcher,enabled and mode=='cpu_copy'))
    with context as metrics:yield metrics

@contextmanager
def retain_mmap(patcher):
    """Retain immutable CPU GGUF mappings; native Comfy owns placement/pinning.

    Support only known pinning semantics. Do not claim mappings were released.
    Scope restores methods on all exits; restart with flag0 for full rollback.
    """
    if type(patcher).__name__!='GGUFModelPatcher':
        yield {'strategy':'native','reason':'non-GGUF patcher'};return
    import comfy.model_management as mm
    allowed=getattr(mm,'PINNING_ALLOWED_TYPES',None)
    if getattr(patcher,'weight_inplace_update',False) or allowed is None or 'GGMLTensor' in allowed:
        yield {'strategy':'native','reason':'unsupported in-place or pinning semantics'};return
    cls=type(patcher);native_load=cls.load;native_pin=cls.pin_weight_to_device
    metrics={'strategy':'mmap','loads':0,'pin_requests':0}
    def load(self,*args,force_patch_weights=False,**kwargs):
        if self is not patcher:return native_load(self,*args,force_patch_weights=force_patch_weights,**kwargs)
        metrics['loads']+=1
        return super(cls,self).load(*args,force_patch_weights=True,**kwargs)
    def pin(self,key):
        if self is not patcher:return native_pin(self,key)
        metrics['pin_requests']+=1
        return super(cls,self).pin_weight_to_device(key)
    cls.load=load;cls.pin_weight_to_device=pin
    try:yield metrics
    finally:cls.load=native_load;cls.pin_weight_to_device=native_pin

@contextmanager
def detach_on_cpu(patcher,enabled=True):
    metrics={'cpu_copied_bytes':0,'tensors':0}
    if not enabled or type(patcher).__name__!='GGUFModelPatcher':
        yield metrics;return
    import comfy.utils
    cls=type(patcher);original=cls.load
    def load(self,*args,**kwargs):
        budget=kwargs.get('lowvram_model_memory',0)
        if (self is patcher and not self.mmap_released and 0<budget<self.model_size()):
            # GGMLTensor.clone() deliberately returns self. Its native .to(...,
            # copy=True) allocates separate bytes and retains quantization metadata.
            for module in self.model.modules():
                for name in ('weight','bias'):
                    value=getattr(module,name,None)
                    if not isinstance(value,torch.Tensor) or value.device.type!='cpu':continue
                    detached=value.to(device='cpu',copy=True)
                    for key,attribute in value.__dict__.items():
                        if key!='patches':setattr(detached,key,attribute)
                    if hasattr(value,'patches'):detached.patches=value.patches.copy()
                    comfy.utils.set_attr_param(module,name,detached)
                    metrics['cpu_copied_bytes']+=value.numel()*value.element_size();metrics['tensors']+=1
            self.mmap_released=True;self.named_modules_to_munmap={}
        return original(self,*args,**kwargs)
    cls.load=load
    try:yield metrics
    finally:cls.load=original
