from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ..core.contracts import RuntimePolicy, MapInfo, ArrayRef, GBuffer, Samples, SampleResult, ModelInfo
from ..core.runtime import Store, fingerprint, object_identity
from ..core.geometry import prepare_geometry, plan_views, validate_uv, mesh_arrays
from ..core.materials import apply_materials, unpack_mr
from ..core.conditions import prepare_reference, encode_conditions, build_tokens, build_workset
from ..core.decode import decode_views


class Mesh:
    def __init__(self):
        self.vertices=torch.tensor([[[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]]])
        self.faces=torch.tensor([[[0,1,2],[0,2,3]]])
        self.uvs=torch.tensor([[[0.,1.],[1.,1.],[1.,0.],[0.,0.]]])
        self.normals=torch.tensor([[[0.,0.,1.]]*4])
        self.material={"metallic_factor":.25,"roughness_factor":.5}
        self.texture=torch.full((1,8,8,3),.2)
        self.metallic_roughness=None
        self.vertex_colors=None
        self.vertex_counts=self.face_counts=None
        self.unlit=False
        self.occlusion_in_mr=False

    def _comfy_cache_tensors(self):
        return (self.vertices,self.faces,self.uvs,self.normals,self.texture)


class FakeVAE:
    latent_channels=128
    def __init__(self):
        self.first_stage_model=object()
        self.calls=0
        self.utx_encoder_id=object_identity(self.first_stage_model)
        self.utx_decoder_hash="fixture"
    def encode(self,image):
        self.calls+=1
        rgb=torch.nn.functional.avg_pool2d(image.permute(0,3,1,2),16)
        return torch.cat((rgb,torch.zeros((1,125,*rgb.shape[-2:]))),dim=1)
    def decode(self,latent):
        return torch.nn.functional.interpolate(latent[:,:3],scale_factor=16,mode="nearest").permute(0,2,3,1)
    def decode_tiled(self,latent,**kwargs):
        return self.decode(latent)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.policy=RuntimePolicy(self.tmp.name,ram_cache_mb=8,system_headroom_mb=0,atlas_tile=64)
        self.store=Store(self.policy)

    def conditions(self):
        context,_=prepare_geometry(Mesh())
        plan=plan_views(context,64,64)
        rgba=np.zeros((64,64,4),np.uint8)
        rgba[24:40,24:40,:3]=(128,64,192)
        rgba[24:40,24:40,3]=255
        n=self.store.put(rgba)
        mask=self.store.put(rgba[...,3]>0)
        depth=self.store.put(np.where(rgba[...,3]>0,1.67,0).astype(np.float32))
        gb=GBuffer(plan,(n,)*6,(depth,)*6,(mask,)*6)
        ref=prepare_reference(torch.ones((1,32,32,3))*.5,None,gb,self.policy)
        vae=FakeVAE()
        cond,_=encode_conditions(vae,ref,gb,self.policy)
        pack,_=build_tokens(cond,self.policy)
        return vae,ref,gb,cond,pack

    def test_cache_atomic_header_budget_and_readonly(self):
        value=np.arange(16,dtype=np.float32)
        ref=self.store.put(value,"array")
        self.assertEqual(ref,self.store.lookup("array"))
        self.assertFalse(ref.read().flags.writeable)
        path=Path(ref.path).with_suffix(".json")
        metadata=json.loads(path.read_text())
        metadata["complete"]=False
        path.write_text(json.dumps(metadata))
        self.assertIsNone(self.store.lookup("array"))
        small=Store(replace(self.policy,disk_budget_gb=1e-9))
        with self.assertRaises(MemoryError):
            small.put(np.zeros(100),"too-large")

    def test_geometry_preservation_and_padded_counts(self):
        mesh=Mesh()
        original=mesh.vertices.clone()
        ctx,_=prepare_geometry(mesh)
        np.testing.assert_allclose(ctx.vertices/ctx.scale+np.asarray(ctx.centre),original[0],atol=1e-6)
        self.assertTrue(torch.equal(mesh.vertices,original))
        validate_uv(ctx)
        mesh.vertices=torch.cat((mesh.vertices,torch.full((1,2,3),float("nan"))),dim=1)
        mesh.faces=torch.cat((mesh.faces,torch.full((1,1,3),999)),dim=1)
        mesh.vertex_counts=torch.tensor([4])
        mesh.face_counts=torch.tensor([2])
        self.assertEqual(len(mesh_arrays(mesh)[0]),4)
        mesh.vertices=mesh.vertices.repeat(2,1,1)
        with self.assertRaises(ValueError):
            mesh_arrays(mesh)

    def test_uv_degeneracy_and_out_of_range(self):
        mesh=Mesh()
        mesh.uvs[0,0]=2
        with self.assertRaises(ValueError):
            validate_uv(prepare_geometry(mesh)[0])
        mesh.uvs.zero_()
        with self.assertRaises(ValueError):
            validate_uv(prepare_geometry(mesh)[0])

    def test_small_uv_degeneracy_is_tolerated_but_large_damage_rejected(self):
        base,_=prepare_geometry(Mesh())
        good=np.repeat(np.array([[0,1,2]],np.int32),10000,axis=0)
        bad=np.repeat(np.array([[0,0,2]],np.int32),8,axis=0)
        with self.assertNoLogs("UltraTex",level="WARNING"):
            self.assertEqual(validate_uv(replace(base,faces=np.concatenate((good,bad)))),8)
        too_many=np.repeat(np.array([[0,0,2]],np.int32),22,axis=0)
        with self.assertRaisesRegex(ValueError,"degenerate UVs"):
            validate_uv(replace(base,faces=np.concatenate((good,too_many))))

    def test_atlas_tolerates_only_tiny_uv_collisions(self):
        from ..core import bake
        from ..core.contracts import GBuffer
        context,_=prepare_geometry(Mesh())
        plan=plan_views(context,64,64)
        dummy=self.store.put(np.zeros((64,64),np.float32))
        gbuffer=GBuffer(plan,(dummy,)*6,(dummy,)*6,(dummy,)*6)
        class FakeRenderer:
            overlaps=5
            def __init__(self,ctx): self.normal=ctx.normals[0]
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def draw(self,w,h,*args,count_only=False):
                if count_only:
                    counts=np.zeros((h,w),np.float32)
                    counts.flat[:self.overlaps]=2
                    return counts
                normals=np.zeros((h,w,4),np.float32)
                normals[...,:3]=self.normal
                positions=np.zeros((h,w,4),np.float32)
                positions[...,3]=1
                return normals,positions
        with patch.object(bake,"Renderer",FakeRenderer):
            with self.assertNoLogs("UltraTex",level="WARNING"):
                atlas,report=bake.prepare_atlas(context,gbuffer,64,self.policy)
            self.assertEqual(report["overlap_texels"],5)
            self.assertEqual(atlas.size,64)
            # Reusing the same UVs after repairing normals must not reuse the old
            # atlas-normal cache. The old cache key omitted normals entirely.
            changed=replace(context,normals=np.tile([0.,1.,0.],(len(context.vertices),1)))
            rebuilt,second=bake.prepare_atlas(changed,gbuffer,64,self.policy)
            self.assertFalse(second['cache_hit'])
            np.testing.assert_allclose(rebuilt.tiles[0].normals.read()[10,10],[0,1,0])
            self.assertNotEqual(atlas.tiles[0].normals.digest,rebuilt.tiles[0].normals.digest)
        other=replace(self.policy,spill_directory=str(Path(self.tmp.name)/"newatlas"))
        FakeRenderer.overlaps=22
        with patch.object(bake,"Renderer",FakeRenderer):
            with self.assertRaisesRegex(ValueError,"Overlapping UVs"):
                bake.prepare_atlas(context,gbuffer,64,other)

    def test_missing_chunk_compatibility_fails_instead_of_silent_fallback(self):
        from ..vendor.optim import chunked_model
        model=types.SimpleNamespace(single_blocks=[object()],double_blocks=[])
        with self.assertRaisesRegex(RuntimeError,"cannot patch"):
            with chunked_model(model,{},512):
                pass

    def test_wrong_adapter_task_rejected_even_after_rename(self):
        from ..core.models import validate_adapter_task, _KNOWN_ADAPTER_TASKS
        for digest,task in _KNOWN_ADAPTER_TASKS.items():
            self.assertEqual(validate_adapter_task("renamed.safetensors",task,digest),(digest,task))
            wrong="mr" if task=="albedo" else "albedo"
            with self.assertRaisesRegex(ValueError,"Adapter/task mismatch"):
                validate_adapter_task("renamed.safetensors",wrong,digest)
        with self.assertRaisesRegex(ValueError,"Adapter/task mismatch"):
            validate_adapter_task("ultratex_flux2_mr.safetensors","albedo","unknown")
        self.assertEqual(validate_adapter_task("custom.safetensors","albedo","unknown"),("unknown",None))

    def test_known_tasks_match_published_metadata_and_override_swapped_names(self):
        from ..core.models import _KNOWN_ADAPTER_TASKS,validate_adapter_task,select_adapter_name
        path=Path(__file__).resolve().parents[1]/"validation/published_adapter_hashes.json"
        metadata=json.loads(path.read_text())
        for artifact in metadata['files']:
            task='mr' if artifact['Path'].startswith('flux2_mr/') else 'albedo'
            self.assertEqual(_KNOWN_ADAPTER_TASKS[artifact['Sha256']],task)
            swapped='ultratex_flux2.safetensors' if task=='mr' else 'ultratex_flux2_mr.safetensors'
            self.assertEqual(validate_adapter_task(swapped,task,artifact['Sha256'])[1],task)
        names=['unrelated.safetensors','ultratex_flux2.safetensors','ultratex_flux2_mr.safetensors']
        hashes={names[1]:next(k for k,v in _KNOWN_ADAPTER_TASKS.items() if v=='mr'),
                names[2]:next(k for k,v in _KNOWN_ADAPTER_TASKS.items() if v=='albedo')}
        with patch('comfyui_ultratex_modular.core.models.file_hash',side_effect=lambda name:hashes[name]):
            self.assertEqual(select_adapter_name(names,'albedo',lambda x:x),names[2])
            self.assertEqual(select_adapter_name(names,'mr',lambda x:x),names[1])

    def test_gutters_preserve_surface_and_do_not_seed_from_holes(self):
        from ..core.bake import pad_gutters, bilinear
        image=np.zeros((12,12,3),np.float32)
        cover=np.zeros((12,12),bool)
        cover[3:9,3:9]=True
        known=cover.copy()
        known[5:7,5:7]=False
        image[known]=[.7,.2,.1]
        image[3,3]=0  # An observed black texel is valid evidence.
        original=image.copy()
        dark_edge=bilinear(image,np.array([8.5]),np.array([6.]))
        count=pad_gutters(image,cover,known,2)
        self.assertGreater(count,0)
        np.testing.assert_array_equal(image[cover],original[cover])
        np.testing.assert_array_equal(image[0],original[0])
        np.testing.assert_allclose(image[2,3],0)
        smooth_edge=bilinear(image,np.array([8.5]),np.array([6.]))
        np.testing.assert_allclose(smooth_edge,[[.7,.2,.1]],atol=1e-6)
        self.assertGreater(smooth_edge.sum(),dark_edge.sum())
        empty=np.zeros_like(image)
        self.assertEqual(pad_gutters(empty,cover,np.zeros_like(known),2),0)
        self.assertFalse(empty.any())

    def test_normal_repair_preserves_authored_normals_and_smooths_uv_duplicates(self):
        from ..core.geometry import compute_normals
        # Two perpendicular triangles share geometric positions, but UVs split
        # their indices. Normals must agree on the duplicated edge, not facet.
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[0,0,1],[1,0,0]],np.float32)
        f=np.array([[0,1,2],[3,4,5]],np.int32)
        n=compute_normals(v,f)
        np.testing.assert_allclose(n[0],n[3],atol=1e-6)
        np.testing.assert_allclose(n[1],n[5],atol=1e-6)
        np.testing.assert_allclose(n[0],[0,2**-.5,2**-.5],atol=1e-6)
        authored=np.tile([1.,0.,0.],(6,1)).astype(np.float32)
        authored[2]=np.nan
        repaired=compute_normals(v,f,authored)
        np.testing.assert_array_equal(repaired[[0,1,3,4,5]],authored[[0,1,3,4,5]])
        self.assertTrue(np.isfinite(repaired).all())
        np.testing.assert_allclose(np.linalg.norm(repaired,axis=1),1.,atol=1e-6)

    def test_ram_guard_reclaims_cache_but_does_not_require_unused_two_gib(self):
        from ..core import runtime
        available=types.SimpleNamespace(available=int(2.02*2**30))
        self.store.small(self.store.put(np.ones(100,np.float32)))
        with patch('psutil.virtual_memory',return_value=available),patch.object(runtime.gc,'collect') as collect:
            runtime.ram_preflight(int(.41*2**30),2048)
            collect.assert_called_once()
            self.assertEqual(runtime._CACHE_BYTES,0)
        with patch('psutil.virtual_memory',return_value=types.SimpleNamespace(available=64*2**20)):
            with self.assertRaisesRegex(MemoryError,'after releasing'):
                runtime.ram_preflight(128*2**20,2048)
        readings=[types.SimpleNamespace(available=64*2**20),types.SimpleNamespace(available=512*2**20)]
        with patch('psutil.virtual_memory',side_effect=readings):
            runtime.ram_preflight(128*2**20,2048)

    def test_occlusion_edges_are_rejected_inside_a_silhouette(self):
        from ..core.bake import projection_ramp,valid_view_mask
        depth=np.ones((64,64),np.float32)
        depth[:,32:]=1.3
        ramp=projection_ramp(depth,np.ones_like(depth),6.)
        self.assertTrue((ramp[:,31:33]==0).all())
        self.assertEqual(ramp[32,16],1.)
        rgb=np.zeros((1024,1024,3),np.uint8)
        mask=np.zeros((1024,1024),bool);mask[100:900,100:900]=True
        rgb[101,500]=255;rgb[500,500]=255
        valid=valid_view_mask(rgb,mask)
        self.assertEqual(valid[101,500],0)
        self.assertEqual(valid[500,500],1)  # Interior white detail survives.

    def test_surface_fill_uses_3d_neighbours_and_preserves_observed_black(self):
        from ..core.bake import repair_atlas
        from ..core.contracts import AtlasTile
        # Blue texels are nearest in UV, red texels are nearest on the surface.
        image=np.zeros((4,4,3),np.float32)
        image[:2]=[0,0,1];image[3]=[1,0,0];image[0,0]=0
        positions=np.zeros((4,4,3),np.float32)
        positions[:2,:,0]=10.;positions[2,:,0]=.01
        holes=np.zeros((4,4),np.float32);holes[2]=1
        chart=self.store.put(np.ones((4,4),np.int32))
        tile=AtlasTile(0,0,self.store.put(positions),None,None)
        atlas=types.SimpleNamespace(chart_ids=chart,tiles=(tile,))
        fixed,mask,_=repair_atlas(image,holes,atlas,'nearest_surface',0)
        np.testing.assert_array_equal(fixed[holes==0],image[holes==0])
        np.testing.assert_allclose(fixed[2],np.tile([1,0,0],(4,1)))
        self.assertTrue((mask[2]==1).all())

    def test_all_six_views_are_previewed_by_default(self):
        from ..core.conditions import views_from_images
        from ..core.decode import inspect_views,contact_sheet
        _,_,gb,_,_=self.conditions()
        collection=views_from_images(torch.rand(6,64,64,3),gb,self.policy)
        images,masks,plan=inspect_views(collection)
        self.assertEqual(tuple(images.shape),(6,64,64,3))
        sheet=contact_sheet(images,plan.names,tile_size=64)
        self.assertEqual(tuple(sheet.shape),(1,176,192,3))

    def test_material_only_inherits_base_and_preserves_factors(self):
        mesh=Mesh()
        mesh.metallic_roughness=torch.ones((1,8,8,3))
        original=mesh.metallic_roughness.clone()
        result,_=apply_materials(mesh,roughness=torch.full((1,4,4,3),.8))
        self.assertIs(result.texture,mesh.texture)
        self.assertTrue(torch.equal(mesh.metallic_roughness,original))
        self.assertTrue(torch.allclose(result.metallic_roughness[...,1],torch.tensor(.8)))
        self.assertTrue(torch.allclose(result.metallic_roughness[...,2],torch.tensor(.25)))
        self.assertEqual(mesh.material["roughness_factor"],.5)
        self.assertEqual(result.material["roughness_factor"],1.)

    def test_mr_channel_order_and_metadata_mismatch(self):
        mr=torch.tensor([[[[0.,.2,.8]]]])
        rough,metal=unpack_mr(mr)
        self.assertAlmostEqual(rough.mean().item(),.2,places=6)
        self.assertAlmostEqual(metal.mean().item(),.8,places=6)
        with self.assertRaises(ValueError):
            apply_materials(Mesh(),roughness=rough,map_infos={"roughness":MapInfo("wrong","roughness",1,1)})

    def test_v_up_converts_result_branch_and_scalar_emissive(self):
        mesh=Mesh()
        expected=mesh.uvs.clone()
        mesh.uvs[...,1]=1-mesh.uvs[...,1]
        before=mesh.uvs.clone()
        result,_=apply_materials(mesh,uv_convention="v_up",emissive=torch.ones((1,4,4,1)))
        torch.testing.assert_close(result.uvs,expected)
        torch.testing.assert_close(mesh.uvs,before)
        self.assertEqual(result.emissive.shape,(1,4,4,3))

    def test_independent_opacity_preserves_and_replaces_factors(self):
        mesh=Mesh()
        mesh.material.update(base_color_factor=[.8,.7,.6,.4],utx_opacity=torch.ones((1,8,8))*.5,
                             utx_alpha_mode="MASK",utx_alpha_cutoff=.3)
        recolored,_=apply_materials(mesh,base_color=torch.ones((1,8,8,3)))
        torch.testing.assert_close(recolored.material["utx_opacity"],torch.full((1,8,8),.2))
        self.assertEqual(recolored.material["utx_alpha_mode"],"MASK")
        replaced,_=apply_materials(mesh,opacity=torch.ones((1,8,8,1))*.8)
        torch.testing.assert_close(replaced.material["utx_opacity"],torch.full((1,8,8),.8))
        self.assertEqual(replaced.material["base_color_factor"],[.8,.7,.6,1.])
        self.assertEqual(mesh.material["base_color_factor"],[.8,.7,.6,.4])

    def test_encode_cache_and_position_ids(self):
        vae,ref,gb,cond,pack=self.conditions()
        before=vae.calls
        encode_conditions(vae,ref,gb,self.policy)
        self.assertEqual(vae.calls,before)
        ids=np.array(pack.position_ids.read()[0])
        self.assertTrue(np.all(ids[:pack.target_count,0]==0))
        self.assertTrue(np.all(ids[pack.target_count:pack.target_count*2,0]==10))
        self.assertTrue(np.all(ids[pack.target_count*2:,0]==20))
        self.assertEqual(pack.grid_h,8)
        self.assertEqual(pack.grid_w,12)
        self.assertLess(pack.target_count,pack.grid_h*pack.grid_w)

    def test_native_qwen_conditioning_metadata(self):
        from ..core.conditions import _conditioning_tensor
        value=torch.zeros((1,512,12))
        result=_conditioning_tensor([[value,{"pooled_output":None,"attention_mask":torch.ones((1,512))}]])
        self.assertEqual(result.shape,(1,512,12))
        with self.assertRaisesRegex(ValueError,"Unsupported conditioning"):
            _conditioning_tensor([[value,{"area":(0,0,1,1)}]])

    def test_disk_backed_decode_and_background_reconstruction(self):
        vae,_,gb,_,pack=self.conditions()
        positive=[[torch.zeros((1,512,2)),{}]]
        jobs=build_workset(pack,positive,None,17,2,"albedo",self.policy)
        self.assertEqual([j.seed for j in jobs.jobs],[17,18])
        values=np.zeros((1,pack.target_count,128),np.float32)
        values[...,:3]=[.25,.5,.75]
        result=SampleResult(jobs.jobs[0],self.store.put(values),{})
        output,report=decode_views(vae,Samples((result,),"fixture"),self.policy,tiled=True)
        self.assertEqual(report["views"],6)
        image=output.sets[0].images[0].read()
        np.testing.assert_array_equal(image[0,0],[255,255,255])
        np.testing.assert_allclose(image[32,32],[64,128,191],atol=1)
        self.assertTrue(all(isinstance(x,ArrayRef) for x in output.sets[0].images))

    def test_view_bundle_round_trip_and_path_boundary(self):
        from ..core.conditions import views_from_images
        from ..core.storage import save_views,load_views
        _,_,gb,_,_=self.conditions()
        views=views_from_images(torch.ones((6,64,64,3))*.3,gb,self.policy)
        manifest=Path(save_views(views,self.tmp.name))
        restored=load_views(manifest)
        self.assertEqual(restored.sets[0].plan,views.sets[0].plan)
        np.testing.assert_array_equal(restored.sets[0].images[0].read(),views.sets[0].images[0].read())
        document=json.loads(manifest.read_text())
        document["views"]["sets"][0]["images"][0]["path"]="../../escape.npy"
        manifest.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError,"escapes"):
            load_views(manifest)
        with self.assertRaisesRegex(ValueError,"output directory"):
            save_views(views,self.tmp.name,".")

    def test_preservation_mask_and_schedule(self):
        from ..core.sampling import schedule
        vae,ref,gb,_,_=self.conditions()
        init=torch.full((6,64,64,3),.5)
        keep=torch.ones((6,64,64))
        cond,_=encode_conditions(vae,ref,gb,self.policy,init,keep)
        tokens,_=build_tokens(cond,self.policy)
        self.assertEqual(tokens.init_tokens.shape,(1,tokens.target_count,128))
        self.assertTrue(np.all(tokens.keep_weights.read()==1))
        with self.assertRaises(ValueError):
            schedule(5,2048,.5,False)
        self.assertEqual(schedule(5,2048,0,True),[0.])
        times=schedule(5,2048,.02,True)
        self.assertEqual(times,[.02,0.])

    def test_gguf_weight_reuse_scoped_and_numerically_equal(self):
        from ..vendor.optim import linear_chunks
        class GGMLLayer(torch.nn.Linear):
            calls=0
            def is_ggml_quantized(self):
                return True
            def cast_bias_weight(self,template):
                self.calls+=1
                return self.weight.detach().clone(),self.bias.detach().clone()
        layer=GGMLLayer(8,12)
        value=torch.randn(1,16,8)
        expected=layer(value)
        with linear_chunks(layer,value) as linear:
            actual=torch.cat([linear(x) for x in value.split(4,dim=1)],dim=1)
        self.assertEqual(layer.calls,1)
        torch.testing.assert_close(expected,actual)
        self.assertTrue(all(cell.cell_contents is None for cell in linear.__closure__))

    def test_mlp_reuses_both_quantized_weights_once_per_call(self):
        from ..vendor.optim import chunked_mlp
        class GGMLLayer(torch.nn.Linear):
            calls=0
            def is_ggml_quantized(self):
                return True
            def cast_bias_weight(self,template):
                self.calls+=1
                return self.weight.detach().clone(),self.bias.detach().clone()
        first,last=GGMLLayer(8,12),GGMLLayer(12,8)
        mlp=torch.nn.Sequential(first,torch.nn.GELU(),last)
        value=torch.randn(1,23,8)
        expected=mlp(value)
        with torch.inference_mode():
            actual=chunked_mlp(mlp,5)(value)
        torch.testing.assert_close(actual,expected)
        self.assertEqual((first.calls,last.calls),(1,1))
        hook=first.register_forward_hook(lambda layer,inputs,result:result+1)
        try:
            torch.testing.assert_close(chunked_mlp(mlp,5)(value),mlp(value))
            self.assertEqual((first.calls,last.calls),(1,1))
        finally:
            hook.remove()

    def test_native_backbone_no_longer_requires_gguf_patcher(self):
        from ..core.models import validate_backbone
        class Flux2:
            pass
        native=Flux2()
        native.diffusion_model=types.SimpleNamespace(hidden_size=4096,patch_size=1,
                                                     params=types.SimpleNamespace(axes_dim=(1,2,3,4)))
        native.latent_format=Flux2()
        model=types.SimpleNamespace(model=native)
        comfy=types.ModuleType("comfy")
        comfy.model_base=types.ModuleType("comfy.model_base")
        comfy.model_base.Flux2=Flux2
        with patch.dict(sys.modules,{"comfy":comfy,"comfy.model_base":comfy.model_base}):
            self.assertIs(validate_backbone(model),native.diffusion_model)
            with self.assertRaisesRegex(ValueError,"requires the GGUF"):
                validate_backbone(model,require_gguf=True)
            native.diffusion_model.hidden_size=3072
            with self.assertRaisesRegex(ValueError,"Backbone dimensions"):
                validate_backbone(model)

    def test_twenty_jobs_request_model_once_and_resume(self):
        from ..core import sampling
        _,_,_,_,pack=self.conditions()
        workset=build_workset(pack,[[torch.zeros((1,512,2)),{}]],None,100,20,"albedo",self.policy)
        requests=[]
        comfy=types.ModuleType("comfy")
        mm=types.ModuleType("comfy.model_management")
        mm.get_torch_device=lambda:torch.device("cpu")
        mm.load_models_gpu=lambda models,**kw:requests.append(models)
        mm.throw_exception_if_processing_interrupted=lambda:None
        comfy.model_management=mm
        model=types.SimpleNamespace(model=types.SimpleNamespace(manual_cast_dtype=torch.float32),
                                    get_attachment=lambda key:ModelInfo("fake","albedo","hash",1.))
        def sample(*args,**kwargs):
            job=args[1]
            return np.full((1,job.pack.target_count,128),job.seed,np.float32),{"seed":job.seed}
        with patch.dict(sys.modules,{"comfy":comfy,"comfy.model_management":mm}), \
             patch.object(sampling,"validate_backbone",lambda _:None),patch.object(sampling,"_sample_job",sample), \
             patch.object(sampling,"_validate_text",lambda _:None):
            output,report=sampling.sample_workset(model,workset,self.policy,backend="dense_debug")
            self.assertEqual(len(output.results),20)
            self.assertEqual(len(requests),1)
            self.assertEqual([x.job.seed for x in output.results],list(range(100,120)))
            self.assertEqual(json.loads(Path(output.manifest_path).read_text())["status"],"complete")
            with patch.object(sampling,"_sample_job",side_effect=AssertionError("must reuse cached samples")):
                resumed,_=sampling.sample_workset(model,workset,self.policy,backend="dense_debug")
            self.assertTrue(all(r.report["cache_hit"] for r in resumed.results))
            self.assertEqual(len(requests),1)  # Fully cached reruns never ask the manager to load.
            interrupted=build_workset(pack,[[torch.zeros((1,512,2)),{}]],None,200,20,"albedo",self.policy)
            def stop_after_three(*args,**kwargs):
                if args[1].seed==203:
                    raise RuntimeError("fixture cancellation")
                return sample(*args,**kwargs)
            with patch.object(sampling,"_sample_job",stop_after_three):
                with self.assertRaisesRegex(RuntimeError,"fixture cancellation"):
                    sampling.sample_workset(model,interrupted,self.policy,backend="dense_debug")
            manifests=[json.loads(p.read_text()) for p in Path(self.tmp.name).glob("workset-*.json")]
            partial=next(m for m in manifests if m["status"]=="interrupted_or_failed")
            self.assertEqual(len(partial["completed"]),3)
            recovered,_=sampling.sample_workset(model,interrupted,self.policy,backend="dense_debug")
            self.assertEqual(len(recovered.results),20)
            self.assertTrue(all(r.report.get("cache_hit") for r in recovered.results[:3]))

    def test_encoder_dimensions_rejected_before_model_loading(self):
        from ..core import sampling
        _,_,_,_,pack=self.conditions()
        workset=build_workset(pack,[[torch.zeros((1,512,2)),{}]],None,1,1,"albedo",self.policy)
        model=types.SimpleNamespace(get_attachment=lambda key:ModelInfo("fake","albedo","hash",1.))
        comfy=types.ModuleType("comfy")
        mm=types.ModuleType("comfy.model_management")
        mm.load_models_gpu=lambda *a,**k:self.fail("Invalid text must fail before loading")
        comfy.model_management=mm
        with patch.dict(sys.modules,{"comfy":comfy,"comfy.model_management":mm}), \
             patch.object(sampling,"validate_backbone",lambda _:None):
            with self.assertRaisesRegex(ValueError,"Qwen3-8B conditioning"):
                sampling.sample_workset(model,workset,self.policy)


if __name__=="__main__":
    unittest.main()
