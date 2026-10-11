from __future__ import annotations

import numpy as np
from PIL import Image

from .contracts import Reference, EncodedConditions, TokenPack, Job, Workset, ViewSet, ViewCollection
from .geometry import GRID_ORDER, numpy_cpu
from .runtime import Store, fingerprint, interrupt, measure, object_identity, tensor_from, ram_preflight


def pixels_u8(value):
    a=numpy_cpu(value)
    if not np.isfinite(a).all():
        raise ValueError("Images/masks must contain finite values")
    return np.clip(a*255+.5,0,255).astype(np.uint8)


def prepare_reference(image,mask,gbuffer,policy,polarity="background",alignment="front",rotation=0.):
    image=numpy_cpu(image)
    if image.ndim!=4 or image.shape[0]!=1 or image.shape[-1] not in (3,4):
        raise ValueError("Reference requires a single RGB/RGBA IMAGE")
    rgb=pixels_u8(image[0])
    h,w=rgb.shape[:2]
    if mask is not None:
        mask=numpy_cpu(mask)
        if mask.shape!=(1,h,w):
            raise ValueError("Reference MASK must match the image and have batch size one")
        alpha=1-mask[0] if polarity=="background" else mask[0]
    elif rgb.shape[-1]==4:
        alpha=rgb[...,3]/255.
    else:
        alpha=np.ones((h,w),np.float32)
    rgba=np.dstack((rgb[...,:3],pixels_u8(alpha)))
    pil=Image.fromarray(rgba).rotate(rotation,resample=Image.Resampling.BICUBIC,expand=True)
    rgba=np.asarray(pil)
    yy,xx=np.nonzero(rgba[...,3]>8)
    if not len(xx):
        raise ValueError("Reference foreground is empty; check mask polarity")
    bounds=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
    cropped=pil.crop(bounds)
    size=gbuffer.plan.sample_size
    if alignment=="front":
        y,x=np.nonzero(gbuffer.masks[0].read())
        if not len(x):
            raise ValueError("Front view is empty; change the geometry orientation")
        ratio=size/gbuffer.plan.render_size
        bx,by=x.min()*ratio,y.min()*ratio
        bw,bh=(x.max()+1-x.min())*ratio,(y.max()+1-y.min())*ratio
    else:
        bx=by=.04*size
        bw=bh=.92*size
    scale=min(bw/cropped.width,bh/cropped.height)
    target=(max(1,round(cropped.width*scale)),max(1,round(cropped.height*scale)))
    offset=(round(bx+(bw-target[0])/2),round(by+(bh-target[1])/2))
    canvas=Image.new("RGBA",(size,size),(0,0,0,0))
    canvas.paste(cropped.resize(target,Image.Resampling.LANCZOS),offset)
    ref=Store(policy).put(np.asarray(canvas))
    transform=dict(rotation=rotation,crop=bounds,output_size=target,offset=offset,alignment=alignment)
    return Reference(ref,fingerprint(ref,transform),transform)


def small_mask(rgba,size):
    # Same 2x2 dilation and anchor as the released UltraTex wrapper.
    m=np.asarray(Image.fromarray(rgba).resize((size,size),Image.Resampling.BILINEAR))[...,3]>0
    d=m.copy()
    d[1:]|=m[:-1]
    d[:,1:]|=m[:,:-1]
    d[1:,1:]|=m[:-1,:-1]
    return d


def grid(tiles):
    return np.concatenate([np.concatenate([tiles[v] for v in row],axis=-1) for row in GRID_ORDER],axis=-2)


def pack_latent(latent):
    # Flux2 VAE already packs spatial 2x2 patches into 128 channels; DiT patch_size is 1.
    return latent.transpose(0,2,3,1).reshape(latent.shape[0],-1,latent.shape[1])


def position_ids(h,w,t=0):
    result=np.zeros((1,h,w,4),np.float32)
    result[...,0]=t
    result[0,...,1]=np.arange(h)[:,None]
    result[0,...,2]=np.arange(w)[None,:]
    return result.reshape(1,h*w,4)


def encode_conditions(vae,reference,gbuffer,policy,init_images=None,keep_masks=None):
    import torch
    if getattr(vae,"latent_channels",None)!=128:
        raise ValueError("Encode Conditions needs a FLUX.2 VAE with 128 packed latent channels")
    size=gbuffer.plan.sample_size
    if reference.rgba.shape[:2]!=(size,size):
        raise ValueError("Reference and view plan have different sample sizes")
    if init_images is not None and tuple(init_images.shape)!=(6,size,size,3):
        raise ValueError("Init views must be six RGB images at the sampling resolution")
    if keep_masks is not None and (init_images is None or tuple(keep_masks.shape)!=(6,size,size)):
        raise ValueError("Preservation masks require six matching init views; 1 means preserve")
    store=Store(policy)
    encoder_id=getattr(vae,"utx_encoder_id",object_identity(vae.first_stage_model))
    normals,masks,init,keep=[],[],[],[]

    def encode(rgb,key):
        def run():
            interrupt()
            ram_preflight(size*size*16,policy.system_headroom_mb)
            pixels=torch.from_numpy(np.ascontiguousarray(rgb.astype(np.float32)/255.))[None]
            latent=vae.encode(pixels).detach().float().cpu().numpy()
            if latent.shape!=(1,128,size//16,size//16):
                raise ValueError(f"Unexpected Flux2 VAE latent shape {latent.shape}")
            return latent
        return store.cached(fingerprint("encode-v2",encoder_id,key,size),run)

    with measure("encode_conditions",size=size) as report,torch.inference_mode():
        for i in range(6):
            interrupt()
            rgba=np.asarray(Image.fromarray(gbuffer.normals[i].read()).resize((size,size),Image.Resampling.BILINEAR))
            a=rgba[...,3:4].astype(np.float32)/255.
            rgb=np.clip(rgba[...,:3]*a+255*(1-a)+.5,0,255).astype(np.uint8)
            normals.append(encode(rgb,gbuffer.normals[i].digest))
            masks.append(store.put(small_mask(gbuffer.normals[i].read(),size//16)))
            if init_images is not None:
                data=pixels_u8(init_images[i])
                init.append(encode(data,fingerprint(data)))
            if keep_masks is not None:
                km=numpy_cpu(keep_masks[i],np.float32)
                keep.append(store.put(np.asarray(Image.fromarray(km).resize((size//16,size//16),Image.Resampling.BOX))))
        rgba=reference.rgba.read()
        a=rgba[...,3:4].astype(np.float32)/255.
        rgb=np.clip(rgba[...,:3]*a+255*(1-a)+.5,0,255).astype(np.uint8)
        ref=encode(rgb,reference.identity)
        bg=encode(np.zeros((size,size,3),np.uint8),"black")
        refmask=store.put(small_mask(rgba,size//16))
        report.update(cache_hits=store.hits,cache_writes=store.writes,encoder_id=encoder_id,
                      latent_shape=normals[0].shape,latent_conversion="Flux2 identity")
    return EncodedConditions(gbuffer,tuple(normals),ref,bg,refmask,tuple(masks),tuple(init),tuple(keep),encoder_id),report


def build_tokens(conditions,policy):
    store=Store(policy)
    with measure("build_foreground_tokens") as report:
        m=grid([x.read() for x in conditions.masks])
        fg=np.flatnonzero(m).astype(np.int64)
        rf=np.flatnonzero(conditions.reference_mask.read()).astype(np.int64)
        if not len(fg) or not len(rf):
            raise ValueError("Target or reference contains no foreground tokens")
        normals=grid([x.read() for x in conditions.normal_tiles])
        reference=conditions.reference.read()
        th,tw=normals.shape[-2:]
        rh,rw=reference.shape[-2:]
        refs=np.concatenate((pack_latent(normals)[:,fg],pack_latent(reference)[:,rf]),axis=1)
        ids=np.concatenate((position_ids(th,tw)[:,fg],position_ids(th,tw,10)[:,fg],position_ids(rh,rw,20)[:,rf]),axis=1)
        initial=store.put(pack_latent(grid([x.read() for x in conditions.init_tiles]))[:,fg]) if conditions.init_tiles else None
        preserve=store.put(grid([x.read() for x in conditions.keep_masks]).reshape(1,-1,1)[:,fg]) if conditions.keep_masks else None
        result=TokenPack(conditions,store.put(fg),store.put(refs),store.put(ids),len(fg),len(rf),th,tw,initial,preserve)
        report.update(target_tokens=len(fg),normal_tokens=len(fg),reference_tokens=len(rf),
                      full_target_tokens=th*tw,foreground_ratio=len(fg)/(th*tw))
    return result,report


def _conditioning_tensor(conditioning):
    if len(conditioning)!=1:
        raise ValueError("UltraTex expects one text CONDITIONING entry; regional/multiple conditions are unsupported")
    tensor,meta=conditioning[0]
    if tensor.ndim!=3 or tensor.shape[0]!=1 or tensor.shape[1]!=512:
        raise ValueError(f"Expected one 512-token Qwen3-8B conditioning, got {tuple(tensor.shape)}")
    # Qwen's encoder already applies its attention mask internally. UltraTex's
    # canonical DiT consumes all 512 returned embeddings, including padded positions.
    unsupported=set(meta)-{"pooled_output","guidance","attention_mask"}
    if unsupported:
        raise ValueError(f"Unsupported conditioning metadata: {sorted(unsupported)}")
    return numpy_cpu(tensor.float(),np.float32)


def build_workset(pack,positive,negative,seed,count,task,policy,previous=None):
    if not 1<=count<=1000:
        raise ValueError("Job count must be between 1 and 1000")
    store=Store(policy)
    pos=store.put(_conditioning_tensor(positive))
    neg=store.put(_conditioning_tensor(negative)) if negative is not None else None
    jobs=list(previous.jobs) if previous else []
    for i in range(count):
        s=(int(seed)+i)%(2**63)
        identity=fingerprint("job-v1",pack.references,pack.position_ids,pack.init_tokens,pack.keep_weights,pos,neg,s,task)
        jobs.append(Job(identity,pack,s,pos,neg,task))
    return Workset(tuple(jobs))


def views_from_images(images,gbuffer,policy,task="external",masks=None):
    images=numpy_cpu(images)
    if images.ndim!=4 or images.shape[0]!=6 or images.shape[-1]!=3 or images.shape[1]!=images.shape[2]:
        raise ValueError("Provide six square RGB views in front,left,back,right,top,bottom order")
    size=images.shape[1]
    if size!=gbuffer.plan.sample_size:
        raise ValueError("External view size must equal Plan Views.sample_size")
    store=Store(policy)
    ims,ms=[],[]
    if masks is not None and tuple(masks.shape)!=tuple(images.shape[:3]):
        raise ValueError("View masks must match all six image dimensions")
    for i in range(6):
        ims.append(store.put(pixels_u8(images[i])))
        if masks is None:
            mask=np.asarray(Image.fromarray(np.uint8(gbuffer.masks[i].read())*255).resize((size,size),Image.Resampling.NEAREST))>0
        else:
            mask=np.clip(numpy_cpu(masks[i],np.float32),0,1)
        ms.append(store.put(mask))
    result=ViewSet(gbuffer.plan,tuple(ims),tuple(ms),task,fingerprint(*ims),gbuffer)
    return ViewCollection((result,))
