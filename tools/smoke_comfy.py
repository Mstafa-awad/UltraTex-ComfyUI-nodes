"""Run with ComfyUI's own Python. No model weights loaded and no installation changes."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--comfy-root",required=True)
parser.add_argument("--report")
parser.add_argument("--sparse",action="store_true")
parser.add_argument("--atlas-size",type=int,default=128)
args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,args.comfy_root)
sys.argv=[sys.argv[0]]  # ComfyUI parses argv on import.

import numpy as np
import torch
from comfy_api.latest import Types

from comfyui_ultratex_modular.extension import UltraTexExtension
from comfyui_ultratex_modular.core.contracts import RuntimePolicy
from comfyui_ultratex_modular.core.geometry import prepare_geometry,plan_views,glb_json,file_to_mesh,FOCAL
from comfyui_ultratex_modular.core.render import render_gbuffers
from comfyui_ultratex_modular.core.bake import prepare_atlas,bake_map,repair_atlas
from comfyui_ultratex_modular.core.conditions import views_from_images
from comfyui_ultratex_modular.core.materials import apply_materials,create_file

report={"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"checks":{}}
start=time.perf_counter()
nodes=asyncio.run(UltraTexExtension().get_node_list())
schemas=[node.define_schema() for node in nodes]
assert len({s.node_id for s in schemas})==len(nodes)
report["checks"]["v3_schemas"]=len(schemas)

mesh=Types.MESH(vertices=torch.tensor([[[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]]]),
                faces=torch.tensor([[[0,1,2],[0,2,3]]]),
                uvs=torch.tensor([[[0.,1.],[1.,1.],[1.,0.],[0.,0.]]]),
                normals=torch.tensor([[[0.,0.,1.]]*4]))
with tempfile.TemporaryDirectory(prefix="utx-smoke-") as tmp:
    atlas_size=args.atlas_size
    policy=RuntimePolicy(tmp,system_headroom_mb=256,atlas_tile=min(512,atlas_size//2))
    ctx,_=prepare_geometry(mesh,policy=policy)
    plan=plan_views(ctx,128,128)
    gb,render_report=render_gbuffers(ctx,plan,policy)
    report["render"]=render_report
    assert gb.masks[0].read().sum()>100
    assert not any(t.is_cuda for t in ctx._comfy_cache_tensors() if isinstance(t,torch.Tensor))
    images=torch.full((6,128,128,3),.5)
    images[...,0]=1.
    views=views_from_images(images,gb,policy,task="albedo")
    atlas,raster_report=prepare_atlas(ctx,gb,atlas_size,policy)
    report["raster"]=raster_report
    output,info,observed,confidence,holes,bake_report=bake_map(atlas,views.sets[0],"base_color",policy)
    report["bake"]=bake_report
    assert observed.mean()>.8,observed.mean()
    np.testing.assert_allclose(output[atlas_size//2,atlas_size//2],[1.,.5,.5],atol=.01)
    # Asymmetric world-coordinate colour ramp catches horizontal/vertical UV flips.
    yy,xx=np.mgrid[:128,:128]
    z=gb.depths[0].read()
    camera=np.stack(((xx/128+.5/128-.5)*z/FOCAL,
                     (.5-yy/128-.5/128)*z/FOCAL,-z),axis=-1)
    inverse=np.linalg.inv(np.asarray(plan.matrices[0]))
    world=camera@inverse[:3,:3].T+inverse[:3,3]
    del z  # Release the Windows mmap before TemporaryDirectory cleanup.
    images[0,...,0]=torch.from_numpy(np.clip(world[...,0]/(2*ctx.scale)+.5,0,1))
    images[0,...,1]=torch.from_numpy(np.clip(world[...,1]/(2*ctx.scale)+.5,0,1))
    images[0,...,2]=.25
    ramp_views=views_from_images(images,gb,policy,task="albedo")
    output,info,*_=bake_map(atlas,ramp_views.sets[0],"base_color",policy,view_indices=(0,))
    lo,hi=atlas_size//4,atlas_size*3//4
    for y,x in ((lo,lo),(lo,hi),(hi,lo),(hi,hi)):
        np.testing.assert_allclose(output[y,x],[(x+.5)/atlas_size,1-(y+.5)/atlas_size,.25],atol=.01)
    report["checks"]["asymmetric_uv_orientation"]=True
    result,_=apply_materials(mesh,base_color=torch.from_numpy(output[None]),map_infos={"base_color":info},
                             opacity=torch.ones((1,atlas_size,atlas_size,3))*.5,alpha_mode="MASK")
    glb=create_file(result)
    document,_=glb_json(glb.get_bytes())
    assert document["materials"][0]["alphaMode"]=="MASK"
    assert document["materials"][0]["alphaCutoff"]==.5
    assert torch.equal(mesh.vertices,result.vertices)
    assert mesh.texture is None
    imported,_=file_to_mesh(glb)
    np.testing.assert_allclose(imported.texture[0,lo,lo,:3].cpu().numpy(),output[lo,lo],atol=.01)
    assert imported.material["utx_alpha_mode"]=="MASK"
    np.testing.assert_allclose(imported.material["utx_opacity"].mean().item(),.5,atol=1/255)
    report["checks"]["native_glb_texture_round_trip"]=True
    report["checks"]["native_mesh_bake_material_export"]=True
    again,cached=prepare_atlas(ctx,gb,atlas_size,policy)
    assert cached["cache_hit"]
    report["checks"]["projection_cache"]=True
    # Duplicate UV occupancy must fail rather than silently overwrite the atlas.
    duplicate=Types.MESH(vertices=mesh.vertices,faces=torch.cat((mesh.faces,mesh.faces),dim=1),uvs=mesh.uvs,normals=mesh.normals)
    context,_=prepare_geometry(duplicate)
    plan2=plan_views(context,64,64)
    gb2,_=render_gbuffers(context,plan2,policy)
    try:
        prepare_atlas(context,gb2,64,policy)
    except ValueError as error:
        assert "Overlapping UV" in str(error)
    else:
        raise AssertionError("Overlapping UVs were not rejected")
    report["checks"]["overlap_rejected"]=True

if args.sparse:
    from comfyui_ultratex_modular.core.sampling import sparse_probe
    error=sparse_probe("cuda:0","bfloat16",torch.__version__)
    report["sparse_probe"]={"passed":error is None,"error":error}
    if error:
        raise RuntimeError(error)
report["wall_seconds"]=time.perf_counter()-start
if args.report:
    Path(args.report).write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
