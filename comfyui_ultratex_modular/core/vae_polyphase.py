"""Research: combine nearest-neighbor 2x upsampling and 3x3 convolution.

The 4x4 stride-two transpose kernel represents the same real-number operation.
Combining coefficients changes floating-point rounding. Not release certified.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

def transpose_weight(w):
    transform=torch.tensor([[0,0,1],[0,1,1],[1,1,0],[1,0,0]],device=w.device,dtype=torch.float32)
    return torch.einsum('ak,ockl,bl->coab',transform,w.float(),transform).to(w.dtype).contiguous()

@contextmanager
def polyphase_upsamples(model):
    import comfy.ops
    changes=[];cache={};counts={'transposed':0,'native':0,'weight_bytes':0}
    for module in model.modules():
        if type(module).__name__!='Upsample' or not module.with_conv or not isinstance(module.conv,torch.nn.Conv2d):continue
        conv=module.conv
        if conv.kernel_size!=(3,3) or conv.stride!=(1,1) or conv.padding!=(1,1) or conv.groups!=1:continue
        original=module.forward
        def forward(x,*args,module=module,conv=conv,original=original,**kwargs):
            if x.ndim!=4 or not x.is_cuda or module.scale_factor!=2.0:
                counts['native']+=1;return original(x,*args,**kwargs)
            with comfy.ops.CastBiasWeightContext(conv,x,offloadable=True) as (weight,bias):
                wt=cache.get(module)
                if wt is None:
                    wt=transpose_weight(weight);cache[module]=wt;counts['weight_bytes']+=wt.numel()*wt.element_size()
                result=F.conv_transpose2d(x,wt,bias,stride=2,padding=1)
            counts['transposed']+=1;return result
        changes.append((module,'forward' in module.__dict__,module.__dict__.get('forward')));module.forward=forward
    try:yield counts
    finally:
        for module,had,value in changes:
            if had:module.forward=value
            else:del module.forward
        cache.clear()
