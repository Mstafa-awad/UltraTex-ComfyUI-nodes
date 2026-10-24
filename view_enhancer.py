"""Additive native UTX 2.5 node. No generation or camera changes."""
from __future__ import annotations
import json
import numpy as np
import torch
from comfy_api.latest import io
from .simple import SETUP, VIEWS, schema, validate_bake_branch


class IntelligentViewEnhancer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXIntelligentViewEnhancer", "2.5 — Intelligent View Enhancer", [
            SETUP.Input("setup"), VIEWS.Input("views"),
            io.Boolean.Input("enabled",default=False,tooltip="Off returns the original view bundle, including after an earlier enhancement."),
            io.Boolean.Input("crop",default=True),
            io.Float.Input("padding_percent",default=3.,min=0.,max=25.,step=.5),
            io.Boolean.Input("square_crop",default=False,tooltip="Off keeps rectangular bounds and uses aspect-preserving letterboxing; camera fitting is unchanged."),
            io.Combo.Input("mask_mode",options=["geometry","BiRefNet mask","hybrid mask","none"],default="geometry",
                           tooltip="Geometry reuses exact generation masks. BiRefNet/hybrid take the native BACKGROUND_REMOVAL model or a foreground MASK batch; geometry protects silhouette bounds."),
            io.Combo.Input("upscaler",options=["None","Comfy upscale model"],default="None"),
            io.Combo.Input("target",options=["Original","2048","4096","8192","Native model scale"],default="Original",
                           tooltip="Longest crop side. Native model scale keeps the Comfy upscale model's actual "
                                   "output pixels without a second resize; recommended for RealESR. "
                                   "This is source size; final texture size is selected in UTX3."),
            io.Float.Input("strength",default=1.,min=0.,max=1.,step=.05,tooltip="Blend AI estimate with resampled source. Zero skips AI entirely."),
            io.Combo.Input("vram_mode",options=["Adaptive","Conservative"],default="Adaptive"),
            io.Int.Input("seed",default=0,min=0,max=0xffffffffffffffff),
            io.Boolean.Input("debug_preview",default=False),
            io.Mask.Input("foreground_masks",optional=True),
            io.BackgroundRemoval.Input("bg_removal_model",optional=True),
            io.Custom("VOSR2_MODEL").Input("vosr2_model",optional=True,advanced=True,lazy=True,
                                         tooltip="Retired compatibility input; not executed by supported enhancement."),
            io.UpscaleModel.Input("upscale_model",optional=True),
        ],[
            VIEWS.Output(display_name="enhanced_views"), VIEWS.Output(display_name="original_views"),
            io.Image.Output(display_name="preview_512"),io.Mask.Output(display_name="foreground_512"),
            io.String.Output(display_name="report"),
        ],"Optional RGB-only stage between UTX2 and UTX3. Preserves original views and calibrated cameras, "
          "stores exact crop/pixel-centre transforms, streams active views, and leaves MR untouched. "
          "Use a connected Comfy upscale checkpoint such as the existing RealESR model. "
          "AI detail may be invented: compare before enabling it for a final asset.")

    @classmethod
    def execute(cls,setup,views,enabled=False,crop=True,padding_percent=3.,square_crop=False,
                mask_mode="geometry",upscaler="None",target="Original",strength=1.,
                vram_mode="Adaptive",seed=0,debug_preview=False,foreground_masks=None,
                vosr2_model=None,upscale_model=None,bg_removal_model=None):
        from .core.enhance import enhance_views,restore_original
        validate_bake_branch(setup,views,"albedo")
        retired=upscaler=='VOSR2'
        if retired:
            upscaler='Comfy upscale model' if upscale_model is not None else 'None'
            if upscaler=='None' and target=='Native model scale':target='Original'
        enhanced,report = enhance_views(setup,views,enabled,crop,padding_percent,square_crop,target,
            upscaler,mask_mode,foreground_masks,strength,vram_mode,None,upscale_model,seed,debug_preview,
            bg_removal_model=bg_removal_model)
        if retired:report['retired_upscaler_migration']=dict(previous='VOSR2',current=upscaler)
        if debug_preview:
            from PIL import Image
            imgs=[];masks=[]
            for vs in enhanced.sets:
                for vi in vs.plan.active_indices:
                    rgb=np.asarray(vs.images[vi].read())
                    if rgb.dtype != np.uint8:
                        rgb=np.clip(rgb*255+.5,0,255).astype(np.uint8)
                    imgs.append(np.array(Image.fromarray(rgb).resize((512,512),Image.Resampling.LANCZOS),np.float32)/255)
                    masks.append(np.array(Image.fromarray(np.asarray(vs.masks[vi].read(),np.float32)).resize((512,512),Image.Resampling.BILINEAR)))
            preview=torch.from_numpy(np.stack(imgs));mask=torch.from_numpy(np.stack(masks))
        else:
            preview=torch.zeros((1,1,1,3));mask=torch.zeros((1,1,1))
        return io.NodeOutput(enhanced,restore_original(views),preview,mask,json.dumps(report,indent=2))

    @classmethod
    def validate_inputs(cls,upscaler):
        return True if upscaler in ('None','Comfy upscale model','VOSR2') else 'Choose None or Comfy upscale model'

    @classmethod
    def check_lazy_status(cls,**kwargs):
        return []  # The retired VOSR2 input never schedules its old model loader.
