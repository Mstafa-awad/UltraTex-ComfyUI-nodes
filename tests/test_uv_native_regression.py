"""CPU regression for standalone UV unwrap (ComfyUI GPU backend simulated)."""
from __future__ import annotations

import importlib
import pathlib
import sys
import types

import numpy as np
import pytest
import torch


def _load_simple(monkeypatch):
    fake_api = types.ModuleType('comfy_api')
    fake_latest = types.ModuleType('comfy_api.latest')
    fake_latest.io = types.SimpleNamespace(ComfyNode=object, Custom=lambda name: name)
    fake_api.latest = fake_latest
    monkeypatch.setitem(sys.modules,'comfy_api',fake_api)
    monkeypatch.setitem(sys.modules,'comfy_api.latest',fake_latest)
    root = pathlib.Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module('comfyui_ultratex_modular.simple')


class FakeMesh:
    def __init__(self):
        self.vertices=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],dtype=torch.float32)
        self.faces=torch.tensor([[[0,1,2]]],dtype=torch.int32)
        self.vertex_counts=torch.tensor([3],dtype=torch.int32)
        self.face_counts=torch.tensor([1],dtype=torch.int32)
        self.normals=torch.tensor([[[0.,0.,1.],[0.,0.,1.],[0.,0.,1.]]],dtype=torch.float32)
        self.uvs=None


def test_comfy_gpu_unwrap_keeps_native_v_down(monkeypatch):
    simple=_load_simple(monkeypatch)
    comfy=types.ModuleType('comfy')
    comfy.__path__=[]
    mm=types.ModuleType('comfy.model_management')
    mm.get_torch_device=lambda: torch.device('cpu')
    comfy.model_management=mm
    extras=types.ModuleType('comfy_extras')
    extras.__path__=[]
    mp=types.ModuleType('comfy_extras.nodes_mesh_postprocess')
    mp._prepare_gpu_mesh_processing=lambda *args:None
    native_uv=np.array([[0.,.2],[.1,.4],[.8,.7]],dtype=np.float32)

    def fake_uv_unwrap(positions,faces,segmenter,resolution,padding,weld):
        assert segmenter == 'pec'
        assert resolution == 2048 and padding == 4 and weld == 0.0
        return torch.tensor([0,1,2]),torch.tensor([[0,1,2]]),torch.from_numpy(native_uv.copy())

    mp._uv_unwrap=fake_uv_unwrap
    extras.nodes_mesh_postprocess=mp
    for key,module in [('comfy',comfy),('comfy.model_management',mm),('comfy_extras',extras),('comfy_extras.nodes_mesh_postprocess',mp)]:
        monkeypatch.setitem(sys.modules,key,module)
    out=simple._unwrap_mesh(FakeMesh(),'comfy_gpu',2048,4)
    np.testing.assert_allclose(out.uvs[0].numpy(),native_uv,atol=1e-7)
    assert out.vertex_counts[0].item()==3 and out.face_counts[0].item()==1


def test_xatlas_unwrap_converts_v_up_to_native_v_down(monkeypatch):
    simple=_load_simple(monkeypatch)
    xatlas=types.ModuleType('xatlas')
    class PackOptions:
        pass
    class ChartOptions:
        pass
    class Atlas:
        def add_mesh(self,vertices,faces):
            assert len(vertices)==3
        def generate(self,chart,pack):
            assert pack.resolution==2048 and pack.padding==4 and pack.bilinear
        def __getitem__(self,index):
            assert index==0
            return np.array([0,1,2]),np.array([[0,1,2]]),np.array([[0.,.8],[.1,.6],[.8,.3]],np.float32)
    xatlas.Atlas=Atlas
    xatlas.ChartOptions=ChartOptions
    xatlas.PackOptions=PackOptions
    monkeypatch.setitem(sys.modules,'xatlas',xatlas)
    out=simple._unwrap_mesh(FakeMesh(),'xatlas',2048,4)
    np.testing.assert_allclose(out.uvs[0].numpy(),[[0.,.2],[.1,.4],[.8,.7]],atol=1e-7)


def test_wrapper_contains_no_numpy_asarray_copy_kwarg():
    import ast
    source=pathlib.Path(__file__).resolve().parents[1]/'simple.py'
    tree=ast.parse(source.read_text())
    bad=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='asarray':
            bad.extend(kw for kw in node.keywords if kw.arg=='copy')
    assert not bad
