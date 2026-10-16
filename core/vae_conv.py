"""Research: exact spatial coverage, global normalization, bounded conv workspace."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

def rows_conv(x,w,b,stride,padding,dilation,groups,rows):
    sh,sw=stride;ph,pw=padding;dh,dw=dilation;kh,kw=w.shape[-2:]
    height=(x.shape[-2]+2*ph-dh*(kh-1)-1)//sh+1
    width=(x.shape[-1]+2*pw-dw*(kw-1)-1)//sw+1
    out=torch.empty((x.shape[0],w.shape[0],height,width),device=x.device,dtype=x.dtype)
    for y in range(0,height,rows):
        end=min(height,y+rows);start_i=y*sh-ph;end_i=(end-1)*sh-ph+dh*(kh-1)+1
        patch=x[:,:,max(0,start_i):min(x.shape[-2],end_i),:]
        top=max(0,-start_i);bottom=max(0,end_i-x.shape[-2])
        if top or bottom:patch=F.pad(patch,(0,0,top,bottom))
        part=F.conv2d(patch,w,b,stride,(0,pw),dilation,groups)
        out[:,:,y:end,:]=part
        del part,patch
    return out

@contextmanager
def bounded_convolutions(model,rows=128,minimum_bytes=64*2**20):
    import comfy.ops
    changes=[];counts={'bounded':0,'native':0}
    for module in model.modules():
        if not isinstance(module,torch.nn.Conv2d):continue
        original=module.forward
        def forward(x,module=module,original=original):
            output_bytes=x.shape[0]*module.out_channels*x.shape[-2]*x.shape[-1]*x.element_size()
            if x.ndim!=4 or not x.is_cuda or x.shape[-2]<=rows or output_bytes<minimum_bytes:
                counts['native']+=1;return original(x)
            counts['bounded']+=1
            with comfy.ops.CastBiasWeightContext(module,x,offloadable=True) as (w,b):
                return rows_conv(x,w,b,module.stride,module.padding,module.dilation,module.groups,rows)
        changes.append((module,'forward' in module.__dict__,module.__dict__.get('forward')));module.forward=forward
    try:yield counts
    finally:
        for module,had,value in changes:
            if had:module.forward=value
            else:del module.forward

@contextmanager
def bounded_encode(vae,size):
    original_estimate=vae.memory_used_encode
    changes=[(name,name in vae.__dict__,vae.__dict__.get(name)) for name in ('memory_used_encode','encode_tiled_')]
    def estimate(shape,dtype):
        if shape[0]==1 and tuple(shape[-2:])==(size,size):return 4096*2**20
        return original_estimate(shape,dtype)
    def refuse_tiling(*args,**kwargs):
        raise MemoryError('Layer-wise VAE encoder exceeded its budget; image-level tiling was not substituted. Release GPU memory and retry.')
    vae.memory_used_encode=estimate;vae.encode_tiled_=refuse_tiling
    try:
        with bounded_convolutions(vae.first_stage_model.encoder,rows=128) as counts:yield counts
    finally:
        for name,had,value in changes:
            if had:setattr(vae,name,value)
            else:delattr(vae,name)
