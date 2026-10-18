from __future__ import annotations

import copy
import json
import struct
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import glb_json, mesh_arrays
from .runtime import fingerprint


def image_tensor(image,name):
    if image.ndim!=4 or image.shape[0]!=1 or image.shape[-1] not in (1,3,4):
        raise ValueError(f"{name}: expected one BHWC image with 1, 3, or 4 channels")
    if not torch.isfinite(image).all():
        raise ValueError(f"{name}: nonfinite values")
    return image.detach().to(device="cpu",dtype=torch.float32).clamp(0,1)


def resize_scalar(image,height,width):
    image=image[...,:1]
    if tuple(image.shape[1:3])!=(height,width):
        image=F.interpolate(image.permute(0,3,1,2),size=(height,width),mode="bilinear",align_corners=False).permute(0,2,3,1)
    return image[...,0]


def apply_materials(mesh,normal_convention="OpenGL",alpha_mode="inherit",alpha_cutoff=.5,
                    uv_convention="v_down",map_infos=None,**maps):
    v,faces,uv,source_normals=mesh_arrays(mesh)
    connected={name:image_tensor(value,name) for name,value in maps.items() if value is not None}
    if connected and uv is None:
        raise ValueError("Applying atlas textures requires UVs on the mesh")
    atlas_id=fingerprint("atlas-v1",v,faces,uv,uv_convention) if uv is not None else ""
    for name,info in (map_infos or {}).items():
        if info is not None:
            if info.atlas_id!=atlas_id:
                raise ValueError(f"{name} was baked on different geometry/UVs; transfer or re-bake it first")
            if name in connected and (info.height,info.width)!=tuple(connected[name].shape[1:3]):
                raise ValueError(f"{name} dimensions differ from its metadata")
            if info.semantic!=name:
                raise ValueError(f"{name} socket received metadata tagged {info.semantic}")
    result=copy.copy(mesh)
    from .geometry import compute_normals
    result.normals=torch.zeros_like(mesh.vertices,dtype=torch.float32,device="cpu")
    result.normals[0,:len(v)]=torch.from_numpy(compute_normals(v,faces,source_normals))
    if uv_convention=="v_up":
        # Native ComfyUI MESH/glTF image UVs are v_down. Convert only this result branch.
        result.uvs=mesh.uvs.clone() if mesh.uvs is not None else None
        if result.uvs is not None:
            result.uvs[...,1]=1-result.uvs[...,1]
    result.material=dict(getattr(mesh,"material",None) or {})
    mat=result.material
    if "base_color" in connected:
        result.texture=connected["base_color"].clone()
        if result.texture.shape[-1]==1:
            result.texture=result.texture.expand(-1,-1,-1,3).clone()
        if result.texture.shape[-1]==4:
            mat["utx_opacity"]=result.texture[...,3].clone()
            result.texture=result.texture[...,:3].contiguous()
        elif "opacity" not in connected:
            old_alpha=mat.get("base_color_factor",[1.,1.,1.,1.])[3]
            if "utx_opacity" in mat:
                mat["utx_opacity"]=mat["utx_opacity"]*old_alpha
            elif old_alpha!=1.:
                mat["utx_opacity"]=torch.full(result.texture.shape[:3],float(old_alpha))
        mat["base_color_factor"]=[1.,1.,1.,1.]
        result.vertex_colors=None
    if any(k in connected for k in ("metallic","roughness","occlusion")):
        old=getattr(mesh,"metallic_roughness",None)
        first=next(connected[k] for k in ("roughness","metallic","occlusion") if k in connected)
        h,w=first.shape[1:3]
        packed=torch.ones((1,h,w,3),dtype=torch.float32)
        for name,ch,fallback,factor_key in (("occlusion",0,1.,None),("roughness",1,.7,"roughness_factor"),("metallic",2,0.,"metallic_factor")):
            if name in connected:
                packed[...,ch]=resize_scalar(connected[name],h,w)
            elif old is not None and (ch!=0 or getattr(mesh,"occlusion_in_mr",False)):
                data=image_tensor(old,"existing MR")[...,ch:ch+1]
                factor=mat.get(factor_key,1.) if factor_key else 1.
                factor=1. if factor is None or factor<0 else factor
                packed[...,ch]=resize_scalar(data,h,w)*factor
            else:
                factor=mat.get(factor_key,fallback) if factor_key else fallback
                packed[...,ch]=fallback if factor is None or factor<0 else factor
        result.metallic_roughness=packed
        result.occlusion_in_mr="occlusion" in connected or getattr(mesh,"occlusion_in_mr",False)
        mat["metallic_factor"]=mat["roughness_factor"]=1.
        if "occlusion" in connected:
            mat["occlusion_strength"]=1.
    elif getattr(mesh,"metallic_roughness",None) is None:
        mat.setdefault("metallic_factor",0.)
        mat.setdefault("roughness_factor",.7)
    if "normal" in connected:
        normal=connected["normal"][...,:3].clone()
        if normal.shape[-1]!=3:
            raise ValueError("Normal maps must have three channels")
        if normal_convention=="DirectX":
            normal[...,1]=1-normal[...,1]
        vectors=normal*2-1
        vectors=F.normalize(vectors,dim=-1,eps=1e-8)
        result.normal_map=(vectors+1)*.5
        mat["normal_scale"]=1.
    if "emissive" in connected:
        result.emissive=connected["emissive"][...,:3].clone()
        if result.emissive.shape[-1]==1:
            result.emissive=result.emissive.expand(-1,-1,-1,3).clone()
        mat["emissive_factor"]=[1.,1.,1.]
    if "opacity" in connected:
        if getattr(result,"texture",None) is None:
            h,w=connected["opacity"].shape[1:3]
            result.texture=torch.ones((1,h,w,3))
        texture=image_tensor(result.texture,"base color")[...,:3]
        alpha=resize_scalar(connected["opacity"],*texture.shape[1:3])
        result.texture=texture
        mat["utx_opacity"]=alpha
        factor=mat.get("base_color_factor",[1.,1.,1.,1.])
        mat["base_color_factor"]=[*factor[:3],1.]
    if alpha_mode!="inherit":
        mat["utx_alpha_mode"]=alpha_mode
        mat["utx_alpha_cutoff"]=float(alpha_cutoff)
    result.unlit=False
    return result,{"stage":"apply_materials","replaced":list(connected),"atlas_id":atlas_id,
                   "raw_map_uv_validation":"Only maps with connected metadata can be verified",
                   "opacity_export":"Use UTX Create 3D File to preserve alpha mode/cutoff"}


def unpack_mr(image):
    if image.shape[-1]<3:
        raise ValueError("MR image needs G=roughness and B=metallic")
    return image[...,1:2].expand(-1,-1,-1,3).clone(),image[...,2:3].expand(-1,-1,-1,3).clone()


def create_file(mesh):
    from comfy_api.latest import Types
    from comfy_extras.nodes_save_3d import save_glb, get_mesh_batch_item
    from PIL import Image
    from .geometry import compute_normals
    v,f,_,source_normals=mesh_arrays(mesh)  # reject batches; never silently export only the first asset
    vertices,faces,colors,uv,normals=get_mesh_batch_item(mesh,0)
    normals=torch.from_numpy(compute_normals(v,f,source_normals))
    settings=dict(getattr(mesh,"material",None) or {})
    opacity=settings.pop("utx_opacity",None)
    def png(name):
        value=getattr(mesh,name,None)
        if value is None:
            return None
        value=image_tensor(value,name)
        if name=="texture" and opacity is not None:
            alpha=resize_scalar(opacity[...,None],*value.shape[1:3])
            value=torch.cat((value[...,:3],alpha[...,None]),dim=-1)
        return Image.fromarray(np.clip(value[0].numpy()*255+.5,0,255).astype(np.uint8))
    tangents=getattr(mesh,"tangents",None)
    data=save_glb(vertices,faces,uvs=uv,vertex_colors=colors,normals=normals,
                  texture_image=png("texture"),metallic_roughness_image=png("metallic_roughness"),
                  normal_map_image=png("normal_map"),emissive_image=png("emissive"),
                  tangents=None if tangents is None else tangents[0,:len(vertices)],
                  material=settings,occlusion_in_mr=getattr(mesh,"occlusion_in_mr",False),
                  unlit=getattr(mesh,"unlit",False))
    if not data:
        raise ValueError("Native mesh exporter returned no geometry")
    doc,remainder=glb_json(data)
    for material in doc.get("materials",[]):
        material["alphaMode"]=settings.get("utx_alpha_mode","OPAQUE")
        if material["alphaMode"]=="MASK":
            material["alphaCutoff"]=settings.get("utx_alpha_cutoff",.5)
    encoded=json.dumps(doc,separators=(",",":"),ensure_ascii=True).encode()
    encoded+=b" "*((-len(encoded))%4)
    remaining=data[remainder:]
    result=struct.pack("<4sII",b"glTF",2,20+len(encoded)+len(remaining))+struct.pack("<II",len(encoded),0x4E4F534A)+encoded+remaining
    return Types.File3D(BytesIO(result),file_format="glb")
