"""Unchanged native single-tile execution with scoped CPU feather masks."""
from __future__ import annotations
import itertools
import torch


def tiled_pass(samples, function, tile, overlap, scale, channels, output_device, batch, pbar, masks=None):
    if batch!=1:raise ValueError('Only the validated single-tile schedule is enabled')
    h,w=samples.shape[-2:]
    output=torch.empty((samples.shape[0],channels,round(h*scale),round(w*scale)),device=output_device)
    if masks is None:masks={}
    for bi in range(samples.shape[0]):
        source=samples[bi:bi+1]
        if h<=tile[0] and w<=tile[1]:
            output[bi:bi+1]=function(source).to(output_device)
            pbar.update(1)
            continue
        out=output[bi:bi+1].zero_()
        divisor=torch.zeros((1,1,round(h*scale),round(w*scale)),device=output_device)
        positions=[range(0,size-overlap,t-overlap) if size>t else [0] for size,t in zip((h,w),tile)]
        specs=[]
        for y,x in itertools.product(*positions):
            y=max(0,min(h-overlap,y));x=max(0,min(w-overlap,x))
            specs.append((y,x,min(tile[0],h-y),min(tile[1],w-x)))
        cursor=0
        while cursor<len(specs):
            group=[specs[cursor]]
            inputs=[source[:,:,y:y+th,x:x+tw] for y,x,th,tw in group]
            packed=inputs[0]
            predictions=function(packed).to(output_device)
            for index,(y,x,th,tw) in enumerate(group):
                ps=predictions[index:index+1]
                key=tuple(ps.shape[2:])
                mask=masks.get(key)
                if mask is None:
                    mask=torch.ones((1,1,*key),device=output_device)
                    for dimension in (2,3):
                        feather=round(scale*overlap)
                        if feather>=mask.shape[dimension]:continue
                        for t in range(feather):
                            a=(t+1)/feather
                            mask.narrow(dimension,t,1).mul_(a)
                            mask.narrow(dimension,mask.shape[dimension]-1-t,1).mul_(a)
                    masks[key]=mask
                oy,ox=round(y*scale),round(x*scale)
                oh,ow=min(ps.shape[-2],out.shape[-2]-oy),min(ps.shape[-1],out.shape[-1]-ox)
                out[:,:,oy:oy+oh,ox:ox+ow].add_(ps[:,:,:oh,:ow]*mask[:,:,:oh,:ow])
                divisor[:,:,oy:oy+oh,ox:ox+ow].add_(mask[:,:,:oh,:ow])
                pbar.update(1)
            cursor+=len(group)
            del predictions,packed,inputs
        out.div_(divisor)
    return output


@torch.inference_mode()
def decode_tiled(vae,samples,tile=64,overlap=16,mode='batch1',masks=None):
    import comfy.utils
    import comfy.model_management as mm
    batch=int(mode.removeprefix('batch'))
    vae.throw_exception_if_invalid()
    shape=list(vae._tile_bounded_shape(samples.shape,tile,tile,None));shape[0]=batch
    memory=vae.memory_used_decode(shape,vae.vae_dtype)*batch
    mm.load_models_gpu([vae.patcher],memory_required=memory,force_full_load=vae.disable_offload)
    shapes=((tile*2,tile//2),(tile//2,tile*2),(tile,tile))
    count=sum(comfy.utils.get_tiled_scale_steps(samples.shape[-1],samples.shape[-2],tw,th,overlap) for th,tw in shapes)*samples.shape[0]
    pbar=comfy.utils.ProgressBar(count)
    def decode(x):return vae.first_stage_model.decode(x.to(vae.vae_dtype).to(vae.device)).to(dtype=vae.vae_output_dtype())
    with mm.cuda_device_context(vae.device):
        output=None
        for shape in shapes:
            part=tiled_pass(samples,decode,shape,overlap,vae.upscale_ratio,vae.output_channels,vae.output_device,batch,pbar,masks)
            output=part if output is None else output+part
            del part
        output=vae.process_output(output/3.0)
    return output.movedim(1,-1)
