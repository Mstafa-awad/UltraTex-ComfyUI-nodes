"""Optional RGB enhancement. Camera/depth calibration is never modified."""
from __future__ import annotations

from dataclasses import asdict, replace
import time
import numpy as np

from .contracts import ImageTransform, ViewCollection
from .runtime import Store, fingerprint, interrupt, ram_preflight


def bounds(mask, padding=3., square=True, align_mips=True):
    """Half-open integer bounds, no erosion; keep every foreground pixel."""
    h, w = mask.shape
    yy, xx = np.nonzero(np.asarray(mask) > .01)
    if not len(xx):
        return 0, 0, w, h
    x0, x1, y0, y1 = int(xx.min()), int(xx.max())+1, int(yy.min()), int(yy.max())+1
    pad = max(2, int(np.ceil(max(x1-x0, y1-y0)*float(padding)/100)))
    x0, y0, x1, y1 = max(0,x0-pad), max(0,y0-pad), min(w,x1+pad), min(h,y1+pad)
    # Preserve both 2x mip phases of the original baker. Arbitrary/odd crop
    # origins change avg-pool alignment, and odd canvas sizes omit mip levels.
    # UTX generation dimensions are multiples of 16, so this only expands bounds.
    if align_mips:
        x0,y0=(x0//4)*4,(y0//4)*4
        x1,y1=min(w,((x1+3)//4)*4),min(h,((y1+3)//4)*4)
    if square:
        side = min(max(x1-x0,y1-y0), min(w,h))
        x0 = max(0, min(w-side, ((x0+x1-side)//8)*4 if align_mips else (x0+x1-side)//2))
        y0 = max(0, min(h-side, ((y0+y1-side)//8)*4 if align_mips else (y0+y1-side)//2))
        x1, y1 = x0+side, y0+side
    return x0, y0, x1-x0, y1-y0


def make_transform(shape, box, target="Original"):
    h, w = shape[:2]
    x,y,cw,ch = box
    size = max(cw,ch) if str(target).lower() == "original" else int(target)
    if not 16 <= size <= 8192:
        raise ValueError("Enhancement canvas must be 16..8192 pixels")
    rw, rh = max(1,round(cw*size/max(cw,ch))), max(1,round(ch*size/max(cw,ch)))
    px,py=(size-rw)//2,(size-rh)//2
    if str(target).lower()=="original":
        px,py=(px//4)*4,(py//4)*4
    return ImageTransform(w,h,x,y,cw,ch,size,rw,rh,px,py)


def resample(array, tr, mask=False):
    """One half-pixel resize; masks stay soft and in [0,1]."""
    from PIL import Image
    crop = np.array(array[tr.top:tr.top+tr.height,tr.left:tr.left+tr.width],copy=True)
    if crop.shape[:2] != (tr.resized_height,tr.resized_width):
        if crop.ndim == 3 and crop.dtype != np.uint8:
            crop = np.stack([np.asarray(Image.fromarray(crop[...,c].astype(np.float32)).resize(
                (tr.resized_width,tr.resized_height),Image.Resampling.LANCZOS)) for c in range(crop.shape[2])],-1)
        else:
            crop = np.asarray(Image.fromarray(crop.astype(np.float32) if mask else crop).resize(
                (tr.resized_width,tr.resized_height), Image.Resampling.BILINEAR if mask else Image.Resampling.LANCZOS))
    shape = (tr.output_size,tr.output_size)+array.shape[2:]
    out = np.zeros(shape,np.float32 if mask else crop.dtype)
    if not mask:
        out[...] = 255 if out.dtype == np.uint8 else 1.
    out[tr.pad_top:tr.pad_top+tr.resized_height,tr.pad_left:tr.pad_left+tr.resized_width] = crop
    return np.clip(out,0,1) if mask else out


def restore_original(collection):
    """Zero-copy rollback for both images and masks."""
    if not any(v.sampling for v in collection.sets):
        return collection
    return ViewCollection(tuple(replace(v,images=v.original_images,masks=v.original_masks,
                sampling=(),original_images=(),original_masks=()) if v.sampling else v for v in collection.sets))


def enhance_views(setup, collection, enabled=False, crop=True, padding=3., square=True,
                  target="Original", upscaler="None", mask_mode="geometry", foreground_masks=None,
                  strength=1., vram_mode="Adaptive", vosr_model=None, upscale_model=None,
                  seed=0, debug=False, runner=None, bg_removal_model=None):
    """Accept/output the native UTX_VIEW_COLLECTION; one view resident at a time.

    Optional MASK batch is foreground=1, ordered by active views across rigs.
    BiRefNet runs in its own native segmentation node; this stage consumes its
    mask without guessing private model socket contracts or downloading code.
    """
    start = time.perf_counter()
    report = dict(stage="intelligent_view_enhancer",enabled=bool(enabled),views=[],
                  input_images_preserved=True,geometry_cameras_unchanged=True)
    if not enabled:
        return restore_original(collection), dict(report,seconds=time.perf_counter()-start)
    if not 0 <= strength <= 1 or not 0 <= padding <= 25:
        raise ValueError("Invalid enhancement strength/padding")
    if upscaler not in ("None","VOSR2","Comfy upscale model"):
        raise ValueError("Unknown upscaler")
    if mask_mode not in ("geometry","BiRefNet mask","hybrid mask","none"):
        raise ValueError("Unknown foreground mode")
    if vram_mode not in ("Adaptive","Conservative"):
        raise ValueError("Unknown VRAM mode")
    if any(v.task not in ("albedo","external") for v in collection.sets):
        raise ValueError("UTX 2.5 enhances Base Color RGB only. Connect MR directly to UTX3.")
    collection = restore_original(collection)  # repeated enhancement always starts at the source
    active_count = sum(len(v.plan.active_indices) for v in collection.sets)
    supplied = None
    if foreground_masks is not None:
        supplied = foreground_masks.detach().float().cpu().numpy() if hasattr(foreground_masks,"detach") else np.asarray(foreground_masks)
        if supplied.ndim != 3 or len(supplied) != active_count or not np.isfinite(supplied).all():
            raise ValueError(f"Foreground MASK must have {active_count} finite masks (one per active view)")
    if mask_mode in ("BiRefNet mask","hybrid mask") and supplied is None and bg_removal_model is None:
        raise ValueError("Connect foreground MASKs or BACKGROUND_REMOVAL from ComfyUI's BiRefNet loader")
    if bg_removal_model is not None and getattr(bg_removal_model,"model_type",None) != "birefnet":
        raise ValueError("The optional background-removal model must be BiRefNet")
    if upscaler == "VOSR2" and runner is None:
        from .super_resolution import VOSRRunner
        runner = VOSRRunner(vosr_model,vram_mode)
    if upscaler == "Comfy upscale model" and runner is None:
        from .super_resolution import ComfyUpscaleRunner
        runner = ComfyUpscaleRunner(upscale_model)
    native_target=str(target)=="Native model scale"
    if native_target and upscaler!="Comfy upscale model":
        raise ValueError("Native model scale requires Comfy upscale model and its connected checkpoint")
    store = Store(setup.runtime)
    sets = []; cursor = 0
    for vs in collection.sets:
        images, masks, transforms = list(vs.images),list(vs.masks),[]
        for vi in range(len(vs.images)):
            interrupt()
            ref = vs.images[vi]; h,w = ref.shape[:2]
            if vi not in vs.plan.active_indices:
                transforms.append(ImageTransform(w,h,0,0,w,h,w,w,h))
                continue
            # Drop the previous view's transient arrays before the next RAM
            # preflight. Only disk-backed refs belong to the output bundle.
            base=enhanced=native=result=area=mask=rgb=None
            original_mask = np.asarray(vs.masks[vi].read(),np.float32)
            detection = original_mask
            seg_seconds=0.
            if mask_mode in ("BiRefNet mask","hybrid mask"):
                from PIL import Image
                if supplied is not None:
                    external=supplied[cursor]
                else:
                    import torch
                    raw=ref.read();x=torch.from_numpy(np.array(raw,dtype=np.float32)/(255 if raw.dtype==np.uint8 else 1))[None]
                    seg_start=time.perf_counter()
                    with torch.inference_mode():external=bg_removal_model.encode_image(x)[0].float().cpu().numpy()
                    seg_seconds=time.perf_counter()-seg_start
                    del x
                external = np.asarray(Image.fromarray(external.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))
                # A segmentation failure must never cut a mesh-conditioned appendage.
                detection = np.maximum(original_mask,np.clip(external,0,1))
            if mask_mode == "none":
                detection = np.ones((h,w),np.float32)
            cursor += 1
            box = bounds(detection,padding,square,align_mips=(str(target).lower()=="original" and
                         (upscaler=="None" or strength==0))) if crop else (0,0,w,h)
            if native_target:
                native_scale=getattr(runner,"native_scale",None)
                if native_scale is None or native_scale<=0 or int(native_scale)!=native_scale:
                    raise ValueError("The connected upscale model must report an integer native scale")
                rw,rh=box[2]*int(native_scale),box[3]*int(native_scale)
                side=((max(rw,rh)+15)//16)*16
                if side>8192:raise ValueError("Native model output exceeds the supported 8192px canvas")
                tr=ImageTransform(w,h,*box,side,rw,rh,(side-rw)//2,(side-rh)//2)
            else:
                tr = make_transform(ref.shape,box,target)
            ram_preflight(tr.output_size**2*72,setup.runtime.system_headroom_mb)
            t = time.perf_counter()
            mask = resample(original_mask,tr,True)
            rgb = ref.read()
            # Native strength=1 needs only letterbox storage, not a full
            # resampled source that would be overwritten by the model result.
            if native_target and strength==1:
                base=np.ones((tr.output_size,tr.output_size,3),np.float32)
            else:base = resample(rgb,tr)
            if not np.any(original_mask):
                enhanced = base
                details = {"skipped":"empty foreground"}
            elif upscaler != "None" and strength > 0:
                # Pass native crop + exact target, no ceil(scale) overshoot and second resize.
                native = np.array(rgb[tr.top:tr.top+tr.height,tr.left:tr.left+tr.width])
                result,details = runner(native,(tr.resized_height,tr.resized_width),seed+cursor-1,
                                         original_mask[tr.top:tr.top+tr.height,tr.left:tr.left+tr.width])
                result = np.asarray(result,np.float32)
                if result.shape != (tr.resized_height,tr.resized_width,3) or not np.isfinite(result).all():
                    raise ValueError("Upscaler returned invalid RGB shape/values")
                if native_target and strength==1:
                    enhanced=base
                    area=enhanced[tr.pad_top:tr.pad_top+tr.resized_height,tr.pad_left:tr.pad_left+tr.resized_width]
                    np.clip(result,0,1,out=area)
                else:
                    enhanced = base.astype(np.float32)/(255 if base.dtype==np.uint8 else 1)
                    area = enhanced[tr.pad_top:tr.pad_top+tr.resized_height,tr.pad_left:tr.pad_left+tr.resized_width]
                    area[:] = area*(1-strength)+np.clip(result,0,1)*strength
                    enhanced = np.clip(enhanced,0,1)
            else:
                enhanced,details = base,{"ai":False}
            key = fingerprint("enhance-v1",ref,vs.masks[vi],asdict(tr),upscaler,strength,seed+cursor-1,
                              getattr(runner,"identity",None),fingerprint(enhanced))
            images[vi] = store.put(enhanced,key)
            masks[vi] = store.put(mask)
            transforms.append(tr)
            entry=dict(details)
            entry.update(rig=len(sets),view=vi,transform=asdict(tr),
                foreground_fraction=float(original_mask.mean()),crop_fraction=box[2]*box[3]/(w*h),
                seconds=time.perf_counter()-t,segmentation_seconds=seg_seconds)
            report["views"].append(entry)
            base=enhanced=native=result=area=mask=rgb=None
        sets.append(replace(vs,images=tuple(images),masks=tuple(masks),sampling=tuple(transforms),
                            original_images=vs.images,original_masks=vs.masks))
    report.update(seconds=time.perf_counter()-start,upscaler=upscaler,mask_mode=mask_mode,
                  segmentation_inference=("supplied MASK" if supplied is not None else "native ComfyUI BiRefNet" if mask_mode in ("BiRefNet mask","hybrid mask") else "none; geometry reused"),
                  ai_detail="generated estimate, not recovered ground truth" if upscaler!="None" else "none")
    return ViewCollection(tuple(sets)),report
