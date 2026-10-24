from __future__ import annotations

import numpy as np
import os
from contextlib import nullcontext
import torch
from PIL import Image

from .contracts import ViewSet, ViewCollection
from .geometry import GRID_ORDER
from .runtime import Store, fingerprint, interrupt, measure, object_identity, tensor_from


@torch.inference_mode()
def decode_views(vae,samples,policy,tiled=True,single_layout=False):
    if getattr(vae,"latent_channels",None)!=128:
        raise ValueError("Decode Views requires a Flux2 VAE")
    if not getattr(vae,"utx_decoder_hash",None):
        raise ValueError("Connect VAE through UTX Apply Foreground Decoder for foreground-only reconstruction")
    store=Store(policy)
    from .adaptive import Controller
    from .vae_tiles import decode_tiled as scheduled_decode
    controller=Controller(policy,vae.device) if hasattr(vae,'device') else None
    from .vae_full_context import requested,full_context_session,decode_full
    tile_masks={}
    sets=[]
    from .gguf_load import research_enabled
    from .vae_polyphase import polyphase_upsamples
    polyphase=(not requested(policy) and research_enabled(policy) and os.environ.get('UTX_ARCH_POLYPHASE','0')=='1'
               and type(vae.first_stage_model).__name__=='AutoencoderKL' and getattr(vae,'vae_dtype',None)==torch.bfloat16)
    full_workload=(tiled and bool(samples.results) and all(
        r.job.task=="albedo" and r.job.pack.conditions.gbuffer.plan.sample_size==2048
        and tuple(r.job.pack.conditions.background.shape[-2:])==(128,128) for r in samples.results))
    with measure("decode_views") as report, (polyphase_upsamples(vae.first_stage_model.decoder) if polyphase else nullcontext({})) as decoder_strategy, full_context_session(vae,policy,workload_supported=full_workload) as full_state:
        for result in samples.results:
            interrupt()
            pack=result.job.pack
            conditions=pack.conditions
            if getattr(vae,"utx_encoder_id",object_identity(vae.first_stage_model))!=conditions.encoder_id:
                raise ValueError("Decode VAE must share the encoder used to prepare these conditions")
            fg=tensor_from(pack.foreground_indices).long()
            values=tensor_from(result.foreground)
            bg=tensor_from(conditions.background)
            _,channels,h,w=bg.shape
            # Reconstruct on CPU. Never hold all decoded views in a float32 batch.
            canvas=bg.repeat(1,1,2,3).permute(0,2,3,1).reshape(1,-1,channels).contiguous()
            canvas[:,fg]=values
            canvas=canvas.reshape(1,2*h,3*w,channels).permute(0,3,1,2)
            images=[None]*6
            masks=[None]*6
            size=conditions.gbuffer.plan.sample_size
            active=set(getattr(conditions.gbuffer.plan, "active_indices", tuple(range(6))))
            empty_image=None
            empty_mask=None
            for row,order in enumerate(GRID_ORDER):
                for col,vi in enumerate(order):
                    interrupt()
                    if vi not in active or not np.any(conditions.gbuffer.masks[vi].read()):
                        # No foreground token was generated for this tile.
                        # Do not waste a 2K VAE decode on a redundant view.
                        if empty_image is None:
                            empty_image=store.put(np.full((size,size,3),255,dtype=np.uint8))
                            empty_mask=store.put(np.zeros((size,size),dtype=np.bool_))
                        images[vi]=empty_image
                        masks[vi]=empty_mask
                        continue
                    full=(full_state['enabled'] and not full_state['disabled'] and size==2048 and tuple(bg.shape[-2:])==(128,128))
                    legacy_tag="decode-v3-polyphase1" if polyphase else "decode-v3"
                    key_parts=(result.foreground,pack.foreground_indices,
                                    conditions.background,conditions.gbuffer.plan,
                                    conditions.gbuffer.masks[vi],vae.utx_decoder_hash,
                                    conditions.encoder_id,vi,tiled,policy.decode_tile)
                    if single_layout and tiled:
                        full=False
                        key_parts=key_parts+("single-layout",)
                    key=fingerprint("decode-v4-full-context1" if full else legacy_tag,*key_parts)
                    mask=np.asarray(Image.fromarray(np.uint8(conditions.gbuffer.masks[vi].read())*255).resize((size,size),Image.Resampling.NEAREST))>0
                    masks[vi]=store.put(mask)
                    cached=store.lookup(key)
                    if cached is not None:
                        images[vi]=cached
                        continue
                    tile=canvas[:,:,row*h:(row+1)*h,col*w:(col+1)*w].contiguous()
                    scheduled=(tiled and controller is not None and policy.performance_mode!='original'
                               and type(vae.first_stage_model).__name__=='AutoencoderKL'
                               and not getattr(vae,'handles_tiling',True) and str(vae.output_device)=='cpu'
                               and callable(getattr(vae,'_tile_bounded_shape',None)))
                    if single_layout and tiled:
                        decoded=scheduled_decode(vae,tile,policy.decode_tile,max(8,policy.decode_tile//4),masks=tile_masks,layouts=1)
                    elif full:
                        decoded=decode_full(vae,tile,full_state)
                        if not full_state['last_used_full']:key=fingerprint(legacy_tag,*key_parts)
                    elif scheduled:
                        batch=controller.plan('vae_decode_view')
                        decoded=scheduled_decode(vae,tile,policy.decode_tile,max(8,policy.decode_tile//4),mode=f'batch{batch}',masks=tile_masks)
                    else:
                        decoded=vae.decode_tiled(tile,tile_x=policy.decode_tile,tile_y=policy.decode_tile,
                                                 overlap=max(8,policy.decode_tile//4)) if tiled else vae.decode(tile)
                    if tuple(decoded.shape)!=(1,size,size,3):
                        raise ValueError(f"Decoded shape {tuple(decoded.shape)} is not the requested {size}x{size}")
                    rgb=decoded[0].detach().float().cpu().numpy()
                    rgb[~mask]=1.
                    images[vi]=store.put(np.clip(rgb*255+.5,0,255).astype(np.uint8),key)
                    del decoded,rgb,tile
            sets.append(ViewSet(conditions.gbuffer.plan,tuple(images),tuple(masks),result.job.task,
                                result.job.identity,conditions.gbuffer))
            del canvas,values,bg
        report.update(jobs=len(sets),views=sum(len(getattr(v.plan, "active_indices", tuple(range(6)))) for v in sets),cache_hits=store.hits,storage="uint8 RGB disk arrays",
                      tiled=tiled,decode_tile=policy.decode_tile,
                      adaptive_decisions=controller.records if controller else [],architecture_decoder=decoder_strategy,full_context_decoder=full_state)
    return ViewCollection(tuple(sets)),report


def inspect_views(collection,job_index=0,view_indices=(0,1,2,3,4,5),preview_size=512):
    if not 0<=job_index<len(collection.sets):
        raise ValueError("Job index is outside this view collection")
    if not view_indices or len(view_indices)>6 or any(i<0 or i>5 for i in view_indices):
        raise ValueError("Choose one to six view indices from 0..5")
    views=collection.sets[job_index]
    images,masks=[],[]
    for i in view_indices:
        rgb=views.images[i].read()
        mask=views.masks[i].read()
        # View bundles saved by older versions hold float enhanced RGB; convert only the preview.
        if rgb.dtype != np.uint8:
            rgb=np.clip(rgb*255+.5,0,255).astype(np.uint8)
        if preview_size and rgb.shape[0]>preview_size:
            rgb=np.asarray(Image.fromarray(rgb).resize((preview_size,preview_size),Image.Resampling.LANCZOS))
            mask=np.asarray(Image.fromarray(np.asarray(mask,np.float32)).resize((preview_size,preview_size),Image.Resampling.BILINEAR))
        images.append(rgb.astype(np.float32)/255.)
        masks.append(mask.astype(np.float32))
    return torch.from_numpy(np.stack(images)),torch.from_numpy(np.stack(masks)),views.plan


def contact_sheet(images,labels,tile_size=384):
    """Small labeled overview; never changes the stored full-resolution views."""
    from PIL import ImageDraw
    count=len(images)
    columns=min(3,count)
    rows=(count+columns-1)//columns
    canvas=Image.new("RGB",(columns*tile_size,rows*(tile_size+24)),(28,28,28))
    draw=ImageDraw.Draw(canvas)
    for i,(image,label) in enumerate(zip(images,labels)):
        tile=Image.fromarray(np.uint8(np.clip(image.detach().cpu().numpy()*255+.5,0,255)))
        tile=tile.resize((tile_size,tile_size),Image.Resampling.LANCZOS)
        x=(i%columns)*tile_size;y=(i//columns)*(tile_size+24)
        canvas.paste(tile,(x,y+24));draw.text((x+8,y+5),label,fill=(240,240,240))
    return torch.from_numpy(np.asarray(canvas,dtype=np.float32)[None]/255.)
