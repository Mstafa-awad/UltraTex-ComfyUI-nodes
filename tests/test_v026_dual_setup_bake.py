"""Regression tests for independent Albedo / MR camera setups.

Exercises the actual bake_texture orchestration with a fake raster backend,
so no ComfyUI/GPU/diffusion model is needed. Does not claim visual CUDA parity.
"""
import importlib
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from comfyui_ultratex_modular.core.contracts import AtlasContext, MapInfo


def _load_simple(monkeypatch):
    latest=types.ModuleType('comfy_api.latest')
    class DummyNode: pass
    class Custom:
        def __init__(self,label): self.label=label
        def Input(self,*a,**k): return None
        def Output(self,*a,**k): return None
    latest.io=NS(ComfyNode=DummyNode,Custom=Custom)
    sdk=types.ModuleType('comfy_api')
    monkeypatch.setitem(sys.modules,'comfy_api',sdk)
    monkeypatch.setitem(sys.modules,'comfy_api.latest',latest)
    import comfyui_ultratex_modular.simple as simple
    return simple


def _fixture(branch, views, size, geom='same_geometry', atlas='same_uvs'):
    context=NS(geometry_id=geom, atlas_id=atlas, mesh='native_mesh')
    gbuffers=[];viewsets=[]
    for k in range(views//6):
        p=NS(identity=f'{branch}_camera_{k}', geometry_id=geom, sample_size=size)
        gb=NS(plan=p)
        gbuffers.append(gb)
        viewsets.append(NS(plan=p, gbuffer=gb, task='albedo' if branch=='A' else 'mr'))
    setup=NS(context=context, gbuffers=tuple(gbuffers), runtime=object())
    return setup, NS(sets=tuple(viewsets))


def _mocks(monkeypatch, observations):
    bake=types.ModuleType('comfyui_ultratex_modular.core.bake')
    def prepare_atlas(context,gb,size,runtime):
        observations.append(('atlas_once',size))
        return AtlasContext(context.atlas_id,context.geometry_id,size,(),None,None,gb),{}
    def bake_map(atlas,vs,semantic,policy,**kwargs):
        assert atlas.gbuffer.plan.identity == vs.plan.identity, (semantic,atlas.gbuffer.plan.identity,vs.plan.identity)
        extras=kwargs.get('extra_views',())
        for ev,_ in extras:
            assert ev.plan.geometry_id == vs.plan.geometry_id
        observations.append(('bake',semantic,vs.plan.identity,vs.plan.sample_size,len(extras)))
        array=np.full((8,8,3),0.5,dtype=np.float32)
        mask=np.zeros((8,8),dtype=np.float32)
        return array,MapInfo(atlas.atlas_id,semantic,8,8),mask,mask,mask,{}
    bake.prepare_atlas=prepare_atlas
    bake.bake_map=bake_map
    bake.repair_atlas=lambda image,holes,atlas,mode,pad:(image,None,{})
    materials=types.ModuleType('comfyui_ultratex_modular.core.materials')
    def apply_materials(mesh,map_infos=None,**maps):
        observations.append(('materials',tuple(sorted(maps))))
        assert all(info.atlas_id=='same_uvs' for info in map_infos.values())
        return mesh,{}
    materials.apply_materials=apply_materials
    materials.create_file=lambda mesh:'glb'
    monkeypatch.setitem(sys.modules,bake.__name__,bake)
    monkeypatch.setitem(sys.modules,materials.__name__,materials)


def test_albedo_12_2048_and_mr_6_1024(monkeypatch):
    simple=_load_simple(monkeypatch)
    seen=[]
    _mocks(monkeypatch,seen)
    a_setup,a_views=_fixture('A',12,2048)
    m_setup,m_views=_fixture('M',6,1024)
    out=simple.bake_texture(a_setup,a_views,2048,True,m_setup,m_views)
    assert out[0]=='native_mesh'
    assert [x[:2] for x in seen if x[0]=='bake'] == [('bake','base_color'),('bake','roughness'),('bake','metallic')]
    assert [x[3] for x in seen if x[0]=='bake'] == [2048,1024,1024]
    assert [x[4] for x in seen if x[0]=='bake'] == [1,0,0]
    assert [x for x in seen if x[0]=='atlas_once']==[('atlas_once',2048)]
    assert ('materials',('base_color','metallic','roughness')) in seen


def test_albedo_6_1024_mr_12_1536(monkeypatch):
    simple=_load_simple(monkeypatch)
    seen=[];_mocks(monkeypatch,seen)
    a_setup,a_views=_fixture('A',6,1024)
    m_setup,m_views=_fixture('M',12,1536)
    simple.bake_texture(a_setup,a_views,4096,True,m_setup,m_views)
    assert [x[3] for x in seen if x[0]=='bake']==[1024,1536,1536]
    assert [x[4] for x in seen if x[0]=='bake']==[0,1,1]


def test_rejects_mismatched_uv_atlas(monkeypatch):
    simple=_load_simple(monkeypatch)
    seen=[];_mocks(monkeypatch,seen)
    a_setup,a_views=_fixture('A',12,1024)
    m_setup,m_views=_fixture('M',6,1024,atlas='some_other_uvs')
    with pytest.raises(ValueError,match='different UV layouts'):
        simple.bake_texture(a_setup,a_views,2048,True,m_setup,m_views)
    assert seen==[]


def test_rejects_missing_mr_setup_and_handles_albedo_only(monkeypatch):
    simple=_load_simple(monkeypatch)
    seen=[];_mocks(monkeypatch,seen)
    a_setup,a_views=_fixture('A',6,1024)
    m_setup,m_views=_fixture('M',6,1024)
    with pytest.raises(ValueError,match='Connect BOTH MR setup'):
        simple.bake_texture(a_setup,a_views,2048,True,None,m_views)
    simple.bake_texture(a_setup,a_views,2048,True)
    assert ('materials',('base_color',)) in seen
