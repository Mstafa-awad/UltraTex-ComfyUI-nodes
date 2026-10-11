from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .contracts import ViewSet, ViewCollection
from .geometry import GRID_ORDER
from .runtime import Store, fingerprint, interrupt, measure, object_identity, tensor_from


@torch.inference_mode()
def decode_views(vae,samples,policy,tiled=True):
    if getattr(vae,"latent_channels",None)!=128:
        raise ValueError("Decode Views requires a Flux2 VAE")
    if not getattr(vae,"utx_decoder_hash",None):
        raise ValueError("Connect VAE through UTX Apply Foreground Decoder for foreground-only reconstruction")
    store=Store(policy)
    sets=[]
    with measure("decode_views") as report:
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
            for row,order in enumerate(GRID_ORDER):
                for col,vi in enumerate(order):
                    interrupt()
                    key=fingerprint("decode-v3",result.foreground,pack.foreground_indices,
                                    conditions.background,conditions.gbuffer.plan,
                                    conditions.gbuffer.masks[vi],vae.utx_decoder_hash,
                                    conditions.encoder_id,vi,tiled,policy.decode_tile)
                    mask=np.asarray(Image.fromarray(np.uint8(conditions.gbuffer.masks[vi].read())*255).resize((size,size),Image.Resampling.NEAREST))>0
                    masks[vi]=store.put(mask)
                    cached=store.lookup(key)
                    if cached is not None:
                        images[vi]=cached
                        continue
                    tile=canvas[:,:,row*h:(row+1)*h,col*w:(col+1)*w].contiguous()
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
        report.update(jobs=len(sets),views=len(sets)*6,cache_hits=store.hits,storage="uint8 RGB disk arrays",
                      tiled=tiled,decode_tile=policy.decode_tile)
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
        if preview_size and rgb.shape[0]>preview_size:
            rgb=np.asarray(Image.fromarray(rgb).resize((preview_size,preview_size),Image.Resampling.LANCZOS))
            mask=np.asarray(Image.fromarray(np.uint8(mask)*255).resize((preview_size,preview_size),Image.Resampling.NEAREST))>0
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
