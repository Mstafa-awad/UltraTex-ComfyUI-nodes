from __future__ import annotations

import logging
import os

import numpy as np
from contextlib import nullcontext
from PIL import Image

from .contracts import Reference, EncodedConditions, TokenPack, Job, Workset, ViewSet, ViewCollection
from .geometry import GRID_ORDER, numpy_cpu
from .runtime import Store, fingerprint, interrupt, measure, object_identity, tensor_from, ram_preflight


_LOG=logging.getLogger(__name__)


def _same_latent(a,b):
    """Repeated encodes of one image are bit-identical on a healthy GPU; a corrupted one differs by ~0.1-0.3 on average."""
    return a.shape==b.shape and bool(np.isfinite(a).all() and np.isfinite(b).all()) and float(np.abs(a-b).mean())<=2e-3


def verified_encode(once):
    """ComfyUI's dynamic-VRAM VAE occasionally returns a corrupted latent with no error (seen 23 Oct on an RTX 5050: one
    normal map or 45-degree guide per run, which FLUX then painted as a grey ghost view, and the bad latent stayed cached
    for the whole session). The encoder is deterministic (repeats are bit-identical), so a second encode must match; on a
    mismatch a third one decides. Nothing unverified reaches the cache. UTX_VERIFY_ENCODE=0 skips the check."""
    first=once()
    if os.environ.get('UTX_VERIFY_ENCODE','1')=='0':
        return first
    second=once()
    if _same_latent(first,second):
        return first
    third=once()
    if _same_latent(first,third) or _same_latent(second,third):
        _LOG.warning("[UTX] VAE encode returned two different latents for the same image; a third encode "
                     "confirmed the correct one (corrupted encode discarded).")
        return third
    raise RuntimeError("UltraTex: the VAE encoder gave three different results for the same image. "
                       "GPU memory is unstable; restart ComfyUI (or start it with --disable-dynamic-vram).")


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


# Optional references for the other views of the 0-degree rig, named by what the picture shows. The cameras are named by
# their position: the "right" camera (+X) sees the CHARACTER'S left side, the "left" camera its right side.
EXTRA_VIEWS = {"back": 2, "character_left_side": 3, "character_right_side": 1, "top": 4, "bottom": 5}
VIEW_PICTURE = {2: "back (character seen from behind)", 3: "character's left side (faces image left)",
                1: "character's right side (faces image right)", 4: "top (seen from above)", 5: "bottom (seen from below)"}
ORIENTATION_MARGIN = .03   # another orientation must fit the mesh silhouette clearly better before the picture is turned


def _border_background(rgb, tolerance=24):
    """Background of a picture without alpha or mask: pixels close to the border colour that are connected to the border."""
    from scipy import ndimage
    border=np.concatenate((rgb[0],rgb[-1],rgb[:,0],rgb[:,-1])).astype(np.int16)
    colour=np.median(border,axis=0)
    near=np.abs(rgb.astype(np.int16)-colour).max(-1)<=tolerance
    labels,_=ndimage.label(near)
    edge=np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
    background=np.isin(labels,edge[edge>0])
    return background if .01<background.mean()<.95 else np.zeros(rgb.shape[:2],bool)


def _reference_rgba(image,mask,polarity):
    image=numpy_cpu(image)
    if image.ndim!=4 or image.shape[0]!=1 or image.shape[-1] not in (3,4):
        raise ValueError("Each reference must be a single RGB/RGBA IMAGE")
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
        alpha=(~_border_background(rgb[...,:3])).astype(np.float32)
    return np.dstack((rgb[...,:3],pixels_u8(alpha)))


def _orientations(view_index):
    turns=[(0,False),(0,True)]
    if view_index in (4,5):   # a picture from above/below can be turned any way
        turns+=[(k,m) for k in (1,2,3) for m in (False,True)]
    return turns


def _apply(rgba,turn):
    k,mirror=turn
    out=np.rot90(rgba,k) if k else rgba
    return np.ascontiguousarray(out[:,::-1] if mirror else out)


def _fit(rgba,box,size):
    """Crop to the picture's foreground and fit it centred into box=(x,y,w,h) of a size x size canvas."""
    yy,xx=np.nonzero(rgba[...,3]>8)
    if not len(xx):
        raise ValueError("Reference foreground is empty; check the mask or its polarity")
    cropped=Image.fromarray(rgba).crop((int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)))
    bx,by,bw,bh=box
    scale=min(bw/cropped.width,bh/cropped.height)
    target=(max(1,round(cropped.width*scale)),max(1,round(cropped.height*scale)))
    offset=(round(bx+(bw-target[0])/2),round(by+(bh-target[1])/2))
    canvas=Image.new("RGBA",(size,size),(0,0,0,0))
    canvas.paste(cropped.resize(target,Image.Resampling.LANCZOS),offset)
    return np.asarray(canvas)


def _silhouette_box(view_mask,size):
    y,x=np.nonzero(view_mask)
    if not len(x):
        return None
    ratio=size/view_mask.shape[0]
    return (x.min()*ratio,y.min()*ratio,(x.max()+1-x.min())*ratio,(y.max()+1-y.min())*ratio)


def orient_reference(rgba,view_mask,view_index,probe=256):
    """Pick the turn/mirror of a picture whose foreground best covers the mesh silhouette of its view.

    Front and back pictures of a mirrored silhouette cannot be told apart, so the given orientation is kept unless another
    one fits clearly better (ORIENTATION_MARGIN). Returns (turn, scores) with scores {turn: silhouette IoU}."""
    sil=np.asarray(Image.fromarray(np.uint8(view_mask)*255).resize((probe,probe),Image.Resampling.BOX))>127
    box=_silhouette_box(sil,probe)
    small=np.asarray(Image.fromarray(rgba).resize((max(1,rgba.shape[1]*probe*2//max(rgba.shape[:2])),
                                                    max(1,rgba.shape[0]*probe*2//max(rgba.shape[:2]))),Image.Resampling.BILINEAR))
    scores={}
    for turn in _orientations(view_index):
        placed=_fit(_apply(small,turn),box,probe)[...,3]>127
        scores[turn]=float((placed&sil).sum()/max(1,(placed|sil).sum()))
    best=max(scores,key=scores.get)
    if scores[best]<scores[(0,False)]+ORIENTATION_MARGIN:
        best=(0,False)
    return best,scores


def prepare_view_reference(image,mask,gbuffer,view_index,policy,polarity="background"):
    """A reference picture for one non-front view, fitted onto that view's mesh silhouette.

    Its tokens are later placed on that view's tile of the six-view grid (build_tokens), so FLUX relates the picture to
    that camera the same way it relates the main reference to the front camera. Returns None for an empty view."""
    view_mask=np.asarray(gbuffer.masks[view_index].read())
    if not view_mask.any():
        return None
    rgba=_reference_rgba(image,mask,polarity)
    turn,scores=orient_reference(rgba,view_mask,view_index)
    size=gbuffer.plan.sample_size
    placed=_fit(_apply(rgba,turn),_silhouette_box(view_mask,size),size)
    ref=Store(policy).put(placed)
    transform=dict(view=int(view_index),turn=int(turn[0]),mirrored=bool(turn[1]),silhouette_iou=round(scores[turn],3),
                   given_iou=round(scores[(0,False)],3))
    return Reference(ref,fingerprint("view-reference-v1",ref,transform),transform)


def view_pictures_for(setup,task):
    """Per-view pictures are colour pictures: they guide the albedo pass only. An MR view must start from noise, otherwise
    the picture's green/blue channels are read as roughness/metallic (a red band became roughness ~0, a mirror)."""
    return getattr(setup,"extra_references",()) if task=="albedo" else ()


def view_reference_init(gbuffer,view_references):
    """Start images and hold masks for the first pass: each view with a picture starts from that picture (fitted to its
    silhouette) and its tokens are held to it for the first sampling steps (TokenPack.keep_steps); views without a
    picture are generated freely. Returns (init (6,S,S,3) in 0..1, keep (6,S,S) in 0..1)."""
    size=gbuffer.plan.sample_size
    init=np.ones((6,size,size,3),np.float32)
    keep=np.zeros((6,size,size),np.float32)
    for view_index,ref in view_references:
        rgba=ref.rgba.read().astype(np.float32)/255.
        a=rgba[...,3:4]
        init[view_index]=rgba[...,:3]*a+(1-a)
        silhouette=np.asarray(Image.fromarray(np.uint8(gbuffer.masks[view_index].read())*255).resize((size,size),Image.Resampling.NEAREST))>0
        keep[view_index]=(a[...,0]>.5)&silhouette
    return init,keep


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
    black_latent=None

    def encode(rgb,key):
        def once():
            interrupt()
            from .gguf_load import research_enabled
            bounded=(research_enabled(policy) and size==2048 and getattr(vae,'vae_dtype',None)==torch.bfloat16
                     and getattr(getattr(vae,'device',None),'type',None)=='cuda'
                     and type(vae.first_stage_model).__name__=='AutoencoderKL')
            if (getattr(vae,'device',None) is not None and getattr(vae,'vae_dtype',None) in (torch.bfloat16,torch.float16,torch.float32)
                    and type(vae.first_stage_model).__name__=='AutoencoderKL'):
                from .adaptive import Controller
                controller=Controller(policy,vae.device)
                bounded=bounded and controller.needs_bounded_encode(size,4 if vae.vae_dtype==torch.float32 else 2)
                controller.vae_encode_guard(size,4 if vae.vae_dtype==torch.float32 else 2,bounded=bounded)
            ram_preflight(size*size*16,policy.system_headroom_mb)
            pixels=torch.from_numpy(np.ascontiguousarray(rgb.astype(np.float32)/255.))[None]
            from .vae_conv import bounded_encode
            cuda=torch.cuda.is_available() and getattr(getattr(vae,'device',None),'type',None)=='cuda'
            if cuda:torch.cuda.synchronize(vae.device)
            with bounded_encode(vae,size) if bounded else nullcontext():
                latent=vae.encode(pixels).detach().float().cpu().numpy()
            if cuda:torch.cuda.synchronize(vae.device)
            if latent.shape!=(1,128,size//16,size//16):
                raise ValueError(f"Unexpected Flux2 VAE latent shape {latent.shape}")
            return latent

        cache_key=fingerprint("encode-v3-verified",encoder_id,key,size)
        from .gguf_load import research_enabled
        if research_enabled(policy):
            cache_key=fingerprint("encode-v3-verified-architecture1",encoder_id,key,size,
                                  policy.performance_mode,policy.max_vram_mb,policy.gpu_headroom_mb)
        return store.cached(cache_key,lambda:verified_encode(once))

    with measure("encode_conditions",size=size) as report,torch.inference_mode():
        for i in range(6):
            interrupt()
            # The 10-view diagonal pass occupies four side tiles of the fixed
            # six-tile UltraTex grid. Skip VAE encoding for its two empty tiles.
            if i not in getattr(gbuffer.plan, "active_indices", tuple(range(6))) or not np.any(gbuffer.masks[i].read()):
                if black_latent is None:
                    black_latent=encode(np.zeros((size,size,3),np.uint8),"black")
                normals.append(black_latent)
                masks.append(store.put(np.zeros((size//16,size//16),dtype=np.bool_)))
                if init_images is not None:
                    init.append(black_latent)
                if keep_masks is not None:
                    keep.append(store.put(np.zeros((size//16,size//16),dtype=np.float32)))
                continue
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
        bg=black_latent if black_latent is not None else encode(np.zeros((size,size,3),np.uint8),"black")
        refmask=store.put(small_mask(rgba,size//16))
        report.update(cache_hits=store.hits,cache_writes=store.writes,encoder_id=encoder_id,
                      latent_shape=normals[0].shape,latent_conversion="Flux2 identity")
    return EncodedConditions(gbuffer,tuple(normals),ref,bg,refmask,tuple(masks),tuple(init),tuple(keep),encoder_id),report


def build_tokens(conditions,policy,keep_steps=-1):
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
        result=TokenPack(conditions,store.put(fg),store.put(refs),store.put(ids),len(fg),len(rf),th,tw,initial,preserve,int(keep_steps))
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
        if getattr(pack,"keep_steps",-1)>=0:
            identity=fingerprint(identity,"keep-steps",pack.keep_steps)
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
