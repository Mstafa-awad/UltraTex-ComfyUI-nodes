"""Adapter for a user-selected, locally installed ComfyUI upscale checkpoint.

No model implementation or checkpoint is bundled or downloaded.
"""
from __future__ import annotations
import numpy as np


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
