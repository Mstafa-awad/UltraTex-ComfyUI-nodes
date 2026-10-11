"""Optional local model integration test. Reads existing weights; downloads nothing."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--comfy-root",required=True)
p.add_argument("--report",required=True)
p.add_argument("--sample",action="store_true")
p.add_argument("--size",type=int,default=512)
p.add_argument("--jobs",type=int,default=2)
p.add_argument("--chunk",type=int,default=2048)
p.add_argument("--with-mr",action="store_true")
args=p.parse_args()
root=Path(args.comfy_root)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(root))
sys.argv=[sys.argv[0]]

import numpy as np
import psutil
import torch
import comfy.sd
import comfy.utils
import comfy.model_management as mm
import folder_paths
from comfy_api.latest import Types

from comfyui_ultratex_modular.core.models import foreground_decoder,apply_lora,select_adapter_name
from comfyui_ultratex_modular.core.geometry import prepare_geometry,plan_views
from comfyui_ultratex_modular.core.render import render_gbuffers
from comfyui_ultratex_modular.core.conditions import prepare_reference,encode_conditions,build_tokens,build_workset
from comfyui_ultratex_modular.core.sampling import sample_workset
from comfyui_ultratex_modular.core.decode import decode_views
from comfyui_ultratex_modular.core.contracts import RuntimePolicy

report={"kind":"integration smoke; synthetic text embeddings, not a quality benchmark",
        "size":args.size,"ram_available_before":psutil.virtual_memory().available}
start=time.perf_counter()
adapter_paths={task:folder_paths.get_full_path_or_raise('loras',select_adapter_name(
    folder_paths.get_filename_list('loras'),task,lambda name:folder_paths.get_full_path_or_raise('loras',name)))
    for task in ('albedo','mr')}
sd=comfy.utils.load_torch_file(str(root/"models/vae/flux2-vae.safetensors"),safe_load=True)
vae=comfy.sd.VAE(sd=sd)
del sd
patched,decoder_report=foreground_decoder(vae,root/"models/ultratex/ultratex_flux2_decoder.pt")
report["decoder"]=decoder_report
assert patched.patcher is not vae.patcher
assert patched.first_stage_model is vae.first_stage_model
assert not vae.patcher.patches
assert patched.patcher.patches
report["vae_latent_channels"]=vae.latent_channels

with tempfile.TemporaryDirectory(prefix="utx-model-smoke-") as temp:
    policy=RuntimePolicy(temp,system_headroom_mb=256,token_chunk=args.chunk)
    # A modest foreground footprint, representative of canonical multi-view framing.
    mesh=Types.MESH(vertices=torch.tensor([[[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]]]),
        faces=torch.tensor([[[0,1,2],[0,2,3]]]),uvs=torch.tensor([[[0.,1.],[1.,1.],[1.,0.],[0.,0.]]]),
        normals=torch.tensor([[[0.,0.,1.]]*4]))
    context,_=prepare_geometry(mesh)
    plan=plan_views(context,args.size,args.size)
    gb,_=render_gbuffers(context,plan,policy)
    image=torch.full((1,args.size,args.size,3),.5)
    reference=prepare_reference(image,None,gb,policy)
    conditions,enc_report=encode_conditions(vae,reference,gb,policy)
    tokens,token_report=build_tokens(conditions,policy)
    report["encoding"],report["tokens"]=enc_report,token_report
    if args.sample:
        available=psutil.virtual_memory().available
        if available<7*2**30:
            report["sampling"]={"skipped":"less than 7 GiB system RAM available", "available":available}
        else:
            gguf_dir=root/"custom_nodes/ComfyUI-GGUF"
            spec=importlib.util.spec_from_file_location("utx_test_gguf",gguf_dir/"__init__.py",submodule_search_locations=[str(gguf_dir)])
            module=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=module
            spec.loader.exec_module(module)
            loader=module.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            model=loader.load_unet("flux-2-klein-base-9b-Q4_0.gguf")[0]
            base_model=model
            model,info,lora_report=apply_lora(model,adapter_paths['albedo'],"albedo",1.)
            report["lora"]=lora_report
            dim=model.model.diffusion_model.txt_in.in_features
            text=[[torch.zeros((1,512,dim)),{}]]
            workset=build_workset(tokens,text,None,12345,args.jobs,"albedo",policy)
            mr_model=None
            if args.with_mr:
                mr_model,_,mr_report=apply_lora(base_model,adapter_paths['mr'],"mr",1.)
                report["mr_lora"]=mr_report
                workset=build_workset(tokens,text,None,12345,args.jobs,"mr",policy,workset)
            sampled,sample_report=sample_workset(model,workset,policy,steps=1,guidance=1.,material_model=mr_model)
            report["sampling"]=sample_report
            report["jobs"]=[x.report for x in sampled.results]
            from comfyui_ultratex_modular.vendor.optim import linear_chunks
            layer=model.model.diffusion_model.single_blocks[0].linear1
            x=torch.randn((1,4,layer.in_features),device=mm.get_torch_device(),dtype=torch.bfloat16)
            with torch.inference_mode():
                expected=layer(x)
                with linear_chunks(layer,x) as linear:
                    actual=torch.cat([linear(x[:,:2]),linear(x[:,2:])],dim=1)
            torch.testing.assert_close(expected,actual,atol=.03,rtol=.01)
            report["gguf_weight_reuse"]={"matches_native_linear":True,"max_difference":float((expected-actual).abs().max())}
            del x,expected,actual,layer
            views,dec_report=decode_views(patched,sampled,policy,tiled=True)
            report["decode"]=dec_report
            assert views.sets[0].images[0].shape==(args.size,args.size,3)
            assert sample_report["model_load_requests"]==(2 if args.with_mr else 1)
            assert len(views.sets)==args.jobs*(2 if args.with_mr else 1)
            mm.unload_all_models()
            del sampled,model,views,mr_model,base_model
            gc.collect()
    else:
        # Exercise replacement-patcher loading and sibling restoration without the DiT.
        latent=torch.from_numpy(np.array(conditions.background.read()))
        a=vae.decode_tiled(latent,tile_x=32,tile_y=32,overlap=8)
        b=patched.decode_tiled(latent,tile_x=32,tile_y=32,overlap=8)
        a2=vae.decode_tiled(latent,tile_x=32,tile_y=32,overlap=8)
        torch.testing.assert_close(a,a2,rtol=1e-3,atol=1e-3)
        assert torch.isfinite(b).all()
        report["branch_safe_decoder"]={"base_restored":True,"max_difference":float((a-a2).abs().max())}
report["wall_seconds"]=time.perf_counter()-start
Path(args.report).write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
print(json.dumps(report,indent=2,default=str))
# Dispose patcher clones while ComfyUI modules still exist (before interpreter shutdown).
mm.unload_all_models()
del vae,patched
gc.collect()
