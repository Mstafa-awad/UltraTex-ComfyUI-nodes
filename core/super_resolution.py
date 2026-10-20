"""Read-only adapters for locally installed, separately licensed SR models.

No copied model implementation, network download, global precision changes, or
monkey-patching. VOSR's native 512px DiT tiles, noise field and Gaussian fusion
are retained. Fractional scales resize directly to the final target BEFORE SR.
"""
from __future__ import annotations
import importlib
import importlib.util
import sys
import time
from pathlib import Path
import numpy as np
from .runtime import interrupt

_BUNDLE_CACHE = {}


def release_cached_vosr():
    """Release only automatically loaded UTX-owned bundles after the view stage.

    Keeps a multi-GB model cache from following FLUX into the next job. Connected
    VOSR2_MODEL inputs remain owned by their native loader and are not touched.
    """
    errors=[];count=0
    if _BUNDLE_CACHE:
        import comfy.model_management as mm
        for model in tuple(_BUNDLE_CACHE.values()):
            for patcher in (model.dit_patcher,model.vae_patcher,model.vision_patcher):
                try:
                    mm.unload_model_and_clones(patcher,unload_additional_models=False)
                    count+=1
                except Exception as error:
                    errors.append(str(error))
        _BUNDLE_CACHE.clear()
    return dict(owned_patchers_released=count,errors=errors)


def vosr_modules():
    import folder_paths
    root = Path(folder_paths.__file__).resolve().parent / "custom_nodes" / "ComfyUI-VOSR2"
    if not (root/"loader.py").is_file():
        raise RuntimeError("Install ComfyUI-VOSR2 separately; UTX does not download code or weights")
    # Prefer the actual loaded package rather than constructing duplicate classes.
    for name, module in tuple(sys.modules.items()):
        file = getattr(module,"__file__",None)
        if file and name.endswith(".loader") and Path(file).resolve() == (root/"loader.py").resolve():
            return module,importlib.import_module(name.rsplit(".",1)[0]+".inference")
    name = "_utx_local_vosr2"
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,root/"__init__.py",submodule_search_locations=[str(root)])
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(name,None)
            raise
    return importlib.import_module(name+".loader"),importlib.import_module(name+".inference")


def existing_bundle(loader):
    import folder_paths
    root=Path(folder_paths.models_dir)/"vosr2"/"VOSR2"
    required=[root/"args.json",root/"dinov2_vitl14.safetensors",
              root/"Qwen-Image-vae-2d/config.json",root/"Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors"]
    if any(not p.is_file() for p in required) or loader._find_dit_weight(root) is None:
        raise RuntimeError("VOSR2 local bundle incomplete. Install its separately licensed weights first; UTX never downloads them.")
    key=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in required+[loader._find_dit_weight(root)])
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE.clear()  # one bundle only; no accumulating GB per workflow
        _BUNDLE_CACHE[key]=loader.load_vosr2("VOSR2","bf16")
    return _BUNDLE_CACHE[key]


def tiled_vosr(model, resized, seed, inference, mask=None, batch_size=1, vae_tile=512):
    """Same tiled flow update as the installed backend, optional tile batching.

    Skip DiT only outside the dependency closure of every VAE decode tile that
    can contribute foreground (including color-alignment/interpolation guard).
    The VAE mid-block has attention: a small silhouette halo is NOT sufficient.
    Keep Gaussian weights/noise for all tiles. VAE stays native FP32.
    """
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import maximum_filter
    stages={};t=time.perf_counter()
    def lap(name):
        nonlocal t
        if resized.device.type=="cuda":torch.cuda.synchronize(resized.device)
        now=time.perf_counter();stages[name]=now-t;t=now
    h,w=resized.shape[-2:]
    padded=inference._pad_to_multiple(resized,16)
    latent,mean,std=model.encode_tiled(padded*2-1,vae_tile,64)
    lap("vae_encode")
    _,channels,lh,lw=latent.shape
    tile,overlap=inference._tile_params(512,64,lh,lw)
    if lh<=tile and lw<=tile:
        sq=inference._pad_to_square(latent)
        features=model.vision_features(padded);lap("vision")
        noise=inference._generate_noise(sq.shape[1:],seed,sq.device,sq.dtype)[None]
        sr=model.denoise_one_step(sq,noise,features)[:,:,:lh,:lw];lap("dit")
        total=used=1
    else:
        tv=importlib.import_module(inference.__package__+".tiled_vae")
        all_tiles=[(y,x) for y in tv._make_tile_grid(lh,tile,overlap) for x in tv._make_tile_grid(lw,tile,overlap)]
        support=None
        if mask is not None:
            m=torch.as_tensor(np.array(mask,dtype=np.float32))[None,None]
            m=F.interpolate(m,size=(h,w),mode="bilinear",align_corners=False)[0,0].numpy()
            foreground=maximum_filter(m>.001,size=65)
            foreground=np.pad(foreground,((0,padded.shape[-2]-h),(0,padded.shape[-1]-w)),mode="edge")
            support=np.zeros((lh,lw),bool)
            vt,vo=tv._tile_params(vae_tile,64,lh,lw)
            for y in tv._make_tile_grid(lh,vt,vo):
                for x in tv._make_tile_grid(lw,vt,vo):
                    if foreground[y*8:(y+vt)*8,x*8:(x+vt)*8].any():
                        support[y:y+vt,x:x+vt]=True
        tiles=[(y,x) for y,x in all_tiles if support is None or support[y:y+tile,x:x+tile].any()]
        total,used=len(all_tiles),len(tiles)
        feats={}
        for i in range(0,len(tiles),batch_size):
            interrupt();group=tiles[i:i+batch_size]
            images=torch.cat([padded[:,:,y*8:(y+tile)*8,x*8:(x+tile)*8] for y,x in group])
            features=model.vision_features(images)
            for j,key in enumerate(group):feats[key]=[v[j:j+1] for v in features]
        lap("vision")
        noise=inference._generate_noise(latent.shape[1:],seed,latent.device,latent.dtype)[None]
        velocity=torch.zeros_like(latent);weight=torch.zeros_like(latent)
        g=tv._gaussian_weights(tile,tile,channels,latent.device)
        for y,x in all_tiles:weight[:,:,y:y+tile,x:x+tile]+=g
        for i in range(0,len(tiles),batch_size):
            interrupt();group=tiles[i:i+batch_size]
            inp=torch.cat([torch.cat([latent[:,:,y:y+tile,x:x+tile],noise[:,:,y:y+tile,x:x+tile]],1) for y,x in group])
            features=[torch.cat([feats[key][j] for key in group]) for j in range(len(feats[group[0]]))]
            out=model.dit_velocity(inp,1.,0.,features)
            for j,(y,x) in enumerate(group):velocity[:,:,y:y+tile,x:x+tile]+=out[j:j+1]*g
        sr=noise-velocity/weight.clamp_min(1e-30);lap("dit")
        del feats,velocity,weight,noise
    decoded=model.decode_tiled(sr,mean,std,vae_tile,64)[:,:,:h,:w];lap("vae_decode")
    # Preserve the installed wavelet alignment contract, including its order.
    color=importlib.import_module(inference.__package__+".color")
    out=color.apply_color_alignment((decoded+1)*.5,resized,"wavelet");lap("color_alignment")
    return out,dict(stage_seconds=stages,dit_tiles_total=total,dit_tiles_used=used,batch_size=batch_size,vae_tile=vae_tile)


class VOSRRunner:
    def __init__(self,model=None,vram_mode="Adaptive"):
        self.model=model;self.mode=vram_mode;self.identity="local VOSR2 BF16 DiT/DINO + original FP32 Qwen VAE"

    def __call__(self,rgb,target,seed,mask):
        import torch
        import torch.nn.functional as F
        import comfy.model_management as mm
        loader,inference=vosr_modules()
        if self.model is None:self.model=existing_bundle(loader)
        model=self.model;device=model.dit_patcher.load_device
        if device.type=="cuda":
            # Account for global image/latent/alignment buffers, resident model
            # weights and tile activations before allocating the target tensor.
            free,total=torch.cuda.mem_get_info(device)
            headroom=(1536 if self.mode=="Conservative" else 768)*2**20
            minimum=target[0]*target[1]*100 + 4*2**30
            if minimum > total-headroom:
                raise MemoryError("Requested VOSR2 target exceeds the conservative device budget. Use None at this resolution, or a smaller enhancement target.")
            # Request activation/global-buffer room, not the weight budget a
            # second time: asking for minimum here would evict cached weights
            # unnecessarily before each image.
            mm.free_memory(target[0]*target[1]*72+headroom,device)
        # No FP16 VAE, changed TF32 flags, forced allocator cap or unload-all.
        x=torch.from_numpy(np.array(rgb,dtype=np.float32)/(255 if rgb.dtype==np.uint8 else 1))[None].movedim(-1,1)
        start=time.perf_counter()
        with torch.inference_mode():
            x=F.interpolate(x.to(device),size=target,mode="bicubic",align_corners=False).clamp_(0,1)
            batch=1
            # Keep serial on every device: the measured BF16 batch-two test
            # changed pixels by up to 0.184. More VRAM does not validate that
            # numerical change. Adaptive placement remains ComfyUI's decision.
            try:
                out,stats=tiled_vosr(model,x,seed,inference,mask,batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Keep target/weights; smaller VAE tiles are a documented fallback
                # with different tile context, and are explicitly reported.
                out,stats=tiled_vosr(model,x,seed,inference,mask,1,256)
                stats["fallback"]="CUDA OOM: 256px VAE tiles; compare for tile differences"
            result=out[0].movedim(0,-1).float().cpu().numpy()
        stats.update(ai=True,model="VOSR2",seconds=time.perf_counter()-start,
                     precision="backend model dtype; Qwen VAE unchanged",fractional_scale="direct target pre-resize")
        return result,stats


class ComfyUpscaleRunner:
    def __init__(self,model):
        if model is None:raise ValueError("Connect UPSCALE_MODEL from ComfyUI's Load Upscale Model node")
        self.model=model;self.native_scale=model.scale;self.identity=f"Comfy UPSCALE_MODEL {type(model).__name__} scale={model.scale}"

    def __call__(self,rgb,target,seed,mask):
        import torch
        import torch.nn.functional as F
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        x=torch.from_numpy(np.array(rgb,dtype=np.float32)/(255 if rgb.dtype==np.uint8 else 1))[None]
        with torch.inference_mode():
            # Run at the checkpoint's native scale, then one precise final resize.
            out=ImageUpscaleWithModel.execute(self.model,x)[0]
            out=out.movedim(-1,1)
            resized=tuple(out.shape[-2:])!=tuple(target)
            if resized:out=F.interpolate(out,size=target,mode="bicubic",align_corners=False,antialias=True)
            out=out.clamp_(0,1)
        return out[0].movedim(0,-1).float().cpu().numpy(),dict(ai=True,model=self.identity,
            post_model_resize=resized,license="user-selected checkpoint; no blanket rights claim")
