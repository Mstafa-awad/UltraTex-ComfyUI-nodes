"""ComfyUI V3 registration. Expensive/optional backends are imported only on execution."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import ComfyExtension, io

from .core import contracts as c

CATEGORY="UltraTex Modular"
POLICY=io.Custom("UTX_RUNTIME_POLICY")
CONTEXT=io.Custom("UTX_MESH_CONTEXT")
PLAN=io.Custom("UTX_VIEW_PLAN")
GBUFFER=io.Custom("UTX_GBUFFER")
REFERENCE=io.Custom("UTX_REFERENCE")
INFO=io.Custom("UTX_MODEL_INFO")
CONDITIONS=io.Custom("UTX_ENCODED_CONDITIONS")
TOKENS=io.Custom("UTX_TOKEN_PACK")
WORKSET=io.Custom("UTX_WORKSET")
SAMPLES=io.Custom("UTX_SAMPLED_LATENTS")
VIEWS=io.Custom("UTX_VIEW_COLLECTION")
ATLAS=io.Custom("UTX_ATLAS_CONTEXT")
MAPINFO=io.Custom("UTX_MAP_INFO")
REPORT=io.Custom("UTX_REPORT")


def schema(name,label,inputs,outputs,description="",output=False):
    basic={"UTXFileToMesh","UTXTextEncode","UTXInspectViews","UTXSaveViews","UTXLoadViews"}
    category=CATEGORY if name in basic else CATEGORY+"/Advanced"
    return io.Schema(node_id=name,display_name="UTX "+label,category=category,inputs=inputs,outputs=outputs,
                     description=description,is_output_node=output)


def indices(text):
    try:
        result=tuple(int(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as error:
        raise ValueError("View indices must be comma-separated integers: 0 front,1 left,2 back,3 right,4 top,5 bottom") from error
    if not result or len(set(result))!=len(result) or any(i<0 or i>5 for i in result):
        raise ValueError("Choose distinct view indices between 0 and 5")
    return result


def decoder_names():
    import folder_paths
    folder_paths.add_model_folder_path("ultratex",str(Path(folder_paths.models_dir)/"ultratex"))
    roots,_=folder_paths.folder_names_and_paths["ultratex"]
    folder_paths.folder_names_and_paths["ultratex"]=(roots,folder_paths.supported_pt_extensions)
    return sorted(set(folder_paths.get_filename_list("ultratex")))


class ConfigureRuntime(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXConfigureRuntime","Configure Runtime",[
            io.String.Input("spill_directory",default="",tooltip="Blank: a per-session folder in output/.ultratex_cache, deleted when ComfyUI exits. A path here keeps intermediates across sessions (clear it yourself)."),
            io.Int.Input("ram_cache_mb",default=256,min=0,max=2048,step=64),
            io.Float.Input("disk_budget_gb",default=16.,min=1.,max=1024.),
            io.Int.Input("system_headroom_mb",default=2048,min=256,max=16384,step=256,tooltip="RAM-cache trimming target, not a required extra 2 GiB allocation."),
            io.Int.Input("gpu_headroom_mb",default=768,min=128,max=8192,step=128),
            io.Int.Input("token_chunk",default=4096,min=0,max=32768,step=256,tooltip="0 disables chunking. Smaller chunks can repeat GGUF dequantization; benchmark before tuning."),
            io.Float.Input("activation_factor",default=12.,min=4.,max=64.,step=.5,tooltip="Conservative initial activation estimate. Calibrate against measured GPU peaks."),
            io.Combo.Input("atlas_tile",options=[256,512,1024],default=512),
            io.Combo.Input("decode_tile",options=[32,64,128],default=64),
            io.Combo.Input("performance_mode",options=['adaptive','conservative','original'],default='adaptive',optional=True),
            io.Int.Input("max_vram_mb",default=0,min=0,max=131072,step=256,optional=True,tooltip='0: driver/physical budget. A lower limit also reduces packed-weight residency.'),
            io.Boolean.Input("adaptive_diagnostics",default=False,optional=True)],
            [POLICY.Output(),REPORT.Output()])

    @classmethod
    def execute(cls,spill_directory,ram_cache_mb,disk_budget_gb,system_headroom_mb,gpu_headroom_mb,token_chunk,activation_factor,atlas_tile,decode_tile,performance_mode='adaptive',max_vram_mb=0,adaptive_diagnostics=False):
        import folder_paths
        from .core.runtime import Store,default_spill_directory
        directory=spill_directory.strip() or str(default_spill_directory(folder_paths.get_output_directory()))
        policy=c.RuntimePolicy(str(Path(directory).resolve()),ram_cache_mb,disk_budget_gb,system_headroom_mb,
                               gpu_headroom_mb,token_chunk,activation_factor,int(atlas_tile),int(decode_tile),performance_mode,int(max_vram_mb),bool(adaptive_diagnostics))
        store=Store(policy)
        return io.NodeOutput(policy,{"stage":"configure_runtime","cache_directory":str(store.root),
                                     "ram_cache_mb":ram_cache_mb,"disk_budget_gb":disk_budget_gb})


class InspectMesh(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXInspectMesh","Inspect Mesh",[io.Mesh.Input("mesh")],[io.Mesh.Output(),REPORT.Output()])

    @classmethod
    def execute(cls,mesh):
        from .core.geometry import mesh_arrays
        v,f,uv,n=mesh_arrays(mesh)
        return io.NodeOutput(mesh,{"stage":"inspect_mesh","vertices":len(v),"faces":len(f),"has_uv":uv is not None,
                                   "has_normals":n is not None,"bounds":[v.min(0).tolist(),v.max(0).tolist()]})


class FileToMesh(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXFileToMesh","File to Mesh",[io.MultiType.Input("file",types=[io.File3DGLB,io.File3DAny])],[io.Mesh.Output(),REPORT.Output()],
                      "Checks scene/material limitations before native import. Refuses lossy multi-material/animated conversion.")

    @classmethod
    def execute(cls,file):
        from .core.geometry import file_to_mesh
        return io.NodeOutput(*file_to_mesh(file))


class PrepareGeometry(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXPrepareGeometry","Prepare Geometry",[io.Mesh.Input("mesh"),POLICY.Input("runtime"),
                    io.Combo.Input("uv_convention",options=["v_down","v_up"],default="v_down")],[CONTEXT.Output(),REPORT.Output()],
                    "Native ComfyUI/glTF uses v_down. Geometry is normalized only for conditions; the original mesh remains intact. Use stock Unwrap Mesh UVs explicitly if needed.")

    @classmethod
    def execute(cls,mesh,runtime,uv_convention):
        from .core.geometry import prepare_geometry
        return io.NodeOutput(*prepare_geometry(mesh,uv_convention,runtime))


class PlanViews(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXPlanViews","Plan Views",[CONTEXT.Input("context"),
            io.Int.Input("render_size",default=2048,min=64,max=4096,step=16),
            io.Combo.Input("sample_size",options=[512,1024,2048,4096],default=1024),
            io.Float.Input("yaw",default=0.,min=-180.,max=180.,step=1.),
            io.Float.Input("elevation",default=0.,min=-90.,max=90.,step=1.)],[PLAN.Output()],
            "Six joint canonical views. 4096 generation is experimental and capacity checked; atlas size is configured separately.")

    @classmethod
    def execute(cls,context,render_size,sample_size,yaw,elevation):
        from .core.geometry import plan_views
        return io.NodeOutput(plan_views(context,render_size,int(sample_size),yaw,elevation))


class RenderGBuffers(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXRenderGBuffers","Render G-buffers",[CONTEXT.Input("context"),PLAN.Input("plan"),POLICY.Input("runtime")],
                      [GBUFFER.Output(),REPORT.Output()])

    @classmethod
    def execute(cls,context,plan,runtime):
        from .core.render import render_gbuffers
        return io.NodeOutput(*render_gbuffers(context,plan,runtime))


class InspectGBuffers(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXInspectGBuffers","Inspect G-buffers",[GBUFFER.Input("gbuffer"),
            io.Int.Input("view",default=0,min=0,max=5),io.Int.Input("preview_size",default=512,min=64,max=2048,step=64)],
            [io.Image.Output(display_name="normal_condition"),io.Mask.Output(display_name="foreground")])

    @classmethod
    def execute(cls,gbuffer,view,preview_size):
        rgba=np.asarray(Image.fromarray(gbuffer.normals[view].read()).resize((preview_size,preview_size),Image.Resampling.BILINEAR))
        alpha=rgba[...,3].astype(np.float32)/255.
        rgb=rgba[...,:3].astype(np.float32)/255.*alpha[...,None]+1-alpha[...,None]
        return io.NodeOutput(torch.from_numpy(rgb[None]),torch.from_numpy(alpha[None]))


class PrepareReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXPrepareReference","Prepare Reference",[io.Image.Input("image"),GBUFFER.Input("gbuffer"),POLICY.Input("runtime"),
            io.Combo.Input("mask_polarity",options=["background","foreground"],default="background",tooltip="ComfyUI Load Image MASK: background. Foreground masks: 1 means object."),
            io.Combo.Input("alignment",options=["front","center"],default="front"),
            io.Float.Input("rotation",default=0.,min=-180.,max=180.,step=1.),io.Mask.Input("mask",optional=True)],
            [REFERENCE.Output(),io.Image.Output(display_name="aligned_preview"),io.Mask.Output(display_name="foreground")])

    @classmethod
    def execute(cls,image,gbuffer,runtime,mask_polarity,alignment,rotation,mask=None):
        from .core.conditions import prepare_reference
        ref=prepare_reference(image,mask,gbuffer,runtime,mask_polarity,alignment,rotation)
        rgba=np.asarray(Image.fromarray(ref.rgba.read()).resize((512,512),Image.Resampling.LANCZOS)).astype(np.float32)/255.
        a=rgba[...,3]
        return io.NodeOutput(ref,torch.from_numpy((rgba[...,:3]*a[...,None]+1-a[...,None])[None]),torch.from_numpy(a[None]))


class ApplyTextureLora(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths
        return schema("UTXApplyTextureLora","Apply Texture LoRA",[io.Model.Input("model"),
            io.Combo.Input("lora_name",options=folder_paths.get_filename_list("loras")),
            io.Combo.Input("task",options=["albedo","mr"],default="albedo"),
            io.Float.Input("strength",default=1.,min=0.,max=2.,step=.05)],
            [io.Model.Output(),INFO.Output(),REPORT.Output()],"Use released Klein base 9B UltraTex PEFT weights. Known adapter fingerprints and explicit filenames are checked against albedo/MR before execution.")

    @classmethod
    def validate_inputs(cls,lora_name,task,**kwargs):
        import folder_paths
        from .core.models import validate_adapter_task
        try:
            validate_adapter_task(folder_paths.get_full_path_or_raise("loras",lora_name),task)
        except (ValueError,FileNotFoundError) as error:
            return str(error)
        return True

    @classmethod
    def fingerprint_inputs(cls,lora_name,**kwargs):
        import folder_paths
        from .core.models import file_hash
        return file_hash(folder_paths.get_full_path_or_raise("loras",lora_name))

    @classmethod
    def execute(cls,model,lora_name,task,strength):
        import folder_paths
        from .core.models import apply_lora
        return io.NodeOutput(*apply_lora(model,folder_paths.get_full_path_or_raise("loras",lora_name),task,strength))


class ApplyForegroundDecoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXApplyForegroundDecoder","Apply Foreground Decoder",[io.Vae.Input("vae"),
            io.Combo.Input("decoder_name",options=decoder_names())],[io.Vae.Output(),REPORT.Output()],
            "Place the released FLUX.2 decoder in models/ultratex. Uses branch-local replacement patches, sharing the base VAE.")

    @classmethod
    def fingerprint_inputs(cls,decoder_name,**kwargs):
        import folder_paths
        from .core.models import file_hash
        return file_hash(folder_paths.get_full_path_or_raise("ultratex",decoder_name))

    @classmethod
    def execute(cls,vae,decoder_name):
        import folder_paths
        from .core.models import foreground_decoder
        return io.NodeOutput(*foreground_decoder(vae,folder_paths.get_full_path_or_raise("ultratex",decoder_name)))


class TextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXTextEncode","Text Encode",[io.Clip.Input("clip"),io.String.Input("text",default="",multiline=True)],
                      [io.Conditioning.Output()],"Use the matching Qwen3-8B / flux2 encoder. Empty text matches image-guided UltraTex defaults.")

    @classmethod
    def execute(cls,clip,text):
        from .core.conditions import _conditioning_tensor
        from .core.gguf_load import storage_context,research_enabled
        with storage_context(clip.patcher,research_enabled(),role='text'):
            conditioned=clip.encode_from_tokens_scheduled(clip.tokenize(text))
        _conditioning_tensor(conditioned)
        return io.NodeOutput(conditioned)


class EncodeConditions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXEncodeConditions","Encode Conditions",[io.Vae.Input("vae"),REFERENCE.Input("reference"),
            GBUFFER.Input("gbuffer"),POLICY.Input("runtime"),io.Image.Input("init_views",optional=True),
            io.Mask.Input("preserve_mask",optional=True)],[CONDITIONS.Output(),REPORT.Output()],
            "CPU/disk latent outputs. Init views must be six images at sample size. Preservation mask: 1 preserves the encoded init content.")

    @classmethod
    def execute(cls,vae,reference,gbuffer,runtime,init_views=None,preserve_mask=None):
        from .core.conditions import encode_conditions
        return io.NodeOutput(*encode_conditions(vae,reference,gbuffer,runtime,init_views,preserve_mask))


class BuildTokens(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXBuildForegroundTokens","Build Foreground Tokens",[CONDITIONS.Input("conditions"),POLICY.Input("runtime")],
                      [TOKENS.Output(),REPORT.Output()])

    @classmethod
    def execute(cls,conditions,runtime):
        from .core.conditions import build_tokens
        return io.NodeOutput(*build_tokens(conditions,runtime))


class BuildWorkset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXBuildWorkset","Build Workset",[TOKENS.Input("tokens"),io.Conditioning.Input("positive"),POLICY.Input("runtime"),
            io.Int.Input("seed",default=0,min=0,max=2**63-1,control_after_generate=True),
            io.Int.Input("count",default=1,min=1,max=1000),io.Combo.Input("task",options=["albedo","mr"],default="albedo"),
            io.Conditioning.Input("negative",optional=True),WORKSET.Input("previous",optional=True)],[WORKSET.Output()],
            "One job = six jointly generated views. Appending supports different prepared conditions; seeds are seed+i. No decoded images accumulate.")

    @classmethod
    def execute(cls,tokens,positive,runtime,seed,count,task,negative=None,previous=None):
        from .core.conditions import build_workset
        return io.NodeOutput(build_workset(tokens,positive,negative,seed,count,task,runtime,previous))


class SampleLatents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXSampleLatents","Sample Latents",[io.Model.Input("model"),WORKSET.Input("workset"),POLICY.Input("runtime"),
            io.Int.Input("steps",default=25,min=1,max=100),io.Float.Input("guidance",default=4.,min=0.,max=20.,step=.1),
            io.Float.Input("denoise",default=1.,min=0.,max=1.,step=.01),
            io.Combo.Input("backend",options=["sparse","dense_debug"],default="sparse"),
            io.Float.Input("sparse_retention",default=.2,min=.01,max=1.,step=.01),
            io.Model.Input("material_model",optional=True)],[SAMPLES.Output(),REPORT.Output()],
            "Sampling only. One model-manager request per task group; sequential CFG. Sparse failures are explicit. Optional MR model must share the same backbone loader. Fewer than 10 steps is a preview for undistilled base 9B.")

    @classmethod
    def execute(cls,model,workset,runtime,steps,guidance,denoise,backend,sparse_retention,material_model=None):
        from .core.sampling import sample_workset
        return io.NodeOutput(*sample_workset(model,workset,runtime,steps,guidance,denoise,backend,sparse_retention,material_model))


class DecodeViews(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXDecodeViews","Decode Views",[io.Vae.Input("vae"),SAMPLES.Input("samples"),POLICY.Input("runtime"),
            io.Boolean.Input("tiled",default=True)],[VIEWS.Output(),REPORT.Output()],
            "Reconstructs the black background on CPU and decodes one view at a time. Outputs disk handles; use Inspect Views for IMAGE sockets.")

    @classmethod
    def execute(cls,vae,samples,runtime,tiled):
        from .core.decode import decode_views
        return io.NodeOutput(*decode_views(vae,samples,runtime,tiled))


class InspectViews(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXInspectViews","Inspect / Split Views",[VIEWS.Input("views"),io.Int.Input("job_index",default=0,min=0,max=1000),
            io.String.Input("view_indices",default="0,1,2,3,4,5"),io.Int.Input("preview_size",default=512,min=0,max=2048,step=64)],
            [io.Image.Output(),io.Mask.Output(),PLAN.Output(),io.Image.Output(display_name="contact_sheet")],
            "Shows all six views by default. 0 preview_size gives full resolution. IMAGE is a batch; contact_sheet displays all selected views together.")

    @classmethod
    def execute(cls,views,job_index,view_indices,preview_size):
        from .core.decode import inspect_views,contact_sheet
        images,masks,plan=inspect_views(views,job_index,indices(view_indices),preview_size)
        return io.NodeOutput(images,masks,plan,contact_sheet(images,[plan.names[i] for i in indices(view_indices)]))


class ViewsFromImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXViewsFromImages","Views From Images",[io.Image.Input("images"),GBUFFER.Input("gbuffer"),POLICY.Input("runtime"),
            io.Combo.Input("task",options=["external","albedo","mr","roughness","metallic"],default="external"),
            io.Mask.Input("foreground_masks",optional=True)],[VIEWS.Output()],
            "Six calibrated external views in front,left,back,right,top,bottom order. Use images rendered/edited from this exact G-buffer rig.")

    @classmethod
    def execute(cls,images,gbuffer,runtime,task,foreground_masks=None):
        from .core.conditions import views_from_images
        return io.NodeOutput(views_from_images(images,gbuffer,runtime,task,foreground_masks))


class UnpackMR(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXUnpackMR","Unpack MR",[io.Image.Input("image")],
                      [io.Image.Output(display_name="roughness"),io.Image.Output(display_name="metallic")])

    @classmethod
    def execute(cls,image):
        from .core.materials import unpack_mr
        return io.NodeOutput(*unpack_mr(image))


class ApplyMaterials(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        names=("base_color","roughness","metallic","occlusion","normal","emissive","opacity")
        return schema("UTXApplyMaterials","Apply Materials",[io.Mesh.Input("mesh"),
            io.Combo.Input("normal_convention",options=["OpenGL","DirectX"],default="OpenGL"),
            io.Combo.Input("alpha_mode",options=["inherit","OPAQUE","MASK","BLEND"],default="inherit"),
            io.Float.Input("alpha_cutoff",default=.5,min=0.,max=1.,step=.05),
            io.Combo.Input("uv_convention",options=["v_down","v_up"],default="v_down")]+
            [io.Image.Input(name,optional=True) for name in names]+
            [MAPINFO.Input(name+"_info",optional=True) for name in names],
            [io.Mesh.Output(),REPORT.Output()],"Independent standard IMAGE maps from any producer. Unconnected maps inherit existing materials. Connect metadata to verify the UV association.")

    @classmethod
    def execute(cls,mesh,normal_convention,alpha_mode,alpha_cutoff,uv_convention,**kwargs):
        from .core.materials import apply_materials
        infos={k[:-5]:v for k,v in kwargs.items() if k.endswith("_info")}
        maps={k:v for k,v in kwargs.items() if not k.endswith("_info")}
        return io.NodeOutput(*apply_materials(mesh,normal_convention,alpha_mode,alpha_cutoff,uv_convention,infos,**maps))


class Create3DFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXCreate3DFile","Create 3D File",[io.Mesh.Input("mesh")],[io.File3DGLB.Output()],
            "Uses the native GLB exporter and preserves UTX opacity mode/cutoff. Connect to stock Save 3D / Preview 3D.")

    @classmethod
    def execute(cls,mesh):
        from .core.materials import create_file
        return io.NodeOutput(create_file(mesh))


class SaveViews(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXSaveViews","Save View Bundle",[VIEWS.Input("views"),io.String.Input("prefix",default="ultratex_views")],
                      [io.String.Output(display_name="manifest_path")],
                      "Saves a portable folder of views, masks, depths, and camera metadata. Load it later to bake without any generation models.",output=True)

    @classmethod
    def execute(cls,views,prefix):
        import folder_paths
        from .core.storage import save_views
        return io.NodeOutput(save_views(views,folder_paths.get_output_directory(),prefix))


class LoadViews(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UTXLoadViews","Load View Bundle",[io.String.Input("manifest_path",default=""),
                      io.Int.Input("job_index",default=0,min=0,max=1000)],
                      [VIEWS.Output(),GBUFFER.Output()],"Load a views.utx.json bundle. G-buffer output is taken from the selected job.")

    @classmethod
    def fingerprint_inputs(cls,manifest_path,job_index=0):
        path=Path(manifest_path).expanduser()
        return path.stat().st_mtime_ns if path.is_file() else "missing"

    @classmethod
    def execute(cls,manifest_path,job_index=0):
        from .core.storage import load_views
        views=load_views(manifest_path)
        if not 0<=job_index<len(views.sets):
            raise ValueError("Job index outside the view bundle")
        return io.NodeOutput(views,views.sets[job_index].gbuffer)




class ProbeGGUFLoRAMemory(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths
        return schema(
            "UTXProbeGGUFLoRAMemory", "GGUF + LoRA VRAM Test",
            [io.Model.Input("model", tooltip="Connect the original Unet Loader (GGUF) MODEL, BEFORE any LoRA."),
             io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
             io.Combo.Input("task", options=["albedo", "mr"], default="albedo"),
             io.Float.Input("strength", default=1., min=0.05, max=2., step=0.05),
             io.Int.Input("test_layers", default=3, min=1, max=8)],
            [io.String.Output(display_name="results"),
             io.String.Output(display_name="json_report_path"),
             REPORT.Output()],
            "Diagnostic ONLY: compares the same Q4 GGUF weight with and without LoRA attached, "
            "without diffusion generation. Saves detailed CUDA memory report in output/ultratex_diagnostics.",
            output=True,
        )

    @classmethod
    def execute(cls, model, lora_name, task, strength, test_layers):
        import folder_paths
        from .core.gguf_lora_probe import probe_gguf_lora
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        summary, report_path, report = probe_gguf_lora(
            model, path, task, float(strength), max_layers=int(test_layers),
            output_dir=folder_paths.get_output_directory()
        )
        return io.NodeOutput(summary, report_path, report)


NODES=[ProbeGGUFLoRAMemory,ConfigureRuntime,InspectMesh,FileToMesh,PrepareGeometry,PlanViews,RenderGBuffers,InspectGBuffers,PrepareReference,
       ApplyTextureLora,ApplyForegroundDecoder,TextEncode,EncodeConditions,BuildTokens,BuildWorkset,SampleLatents,
       DecodeViews,InspectViews,ViewsFromImages,UnpackMR,ApplyMaterials,Create3DFile,
       SaveViews,LoadViews]

from .simple import SIMPLE_NODES
from .fast_baker import FAST_BAKER_NODES
from .view_enhancer import IntelligentViewEnhancer
NODES=SIMPLE_NODES+[IntelligentViewEnhancer]+FAST_BAKER_NODES+NODES

# v2: additive standalone mesh loader; all original node and execution logic unchanged.
from .mesh_loader_v2 import LoadMeshV2
NODES = [LoadMeshV2] + NODES


class UltraTexExtension(ComfyExtension):
    async def get_node_list(self):
        return NODES


# UTX 4: standalone, name-scoped temporary cache cleaner and ANY pass-through.
from .utx_cache_cleaner import UTXCacheCleaner
NODES.append(UTXCacheCleaner)
