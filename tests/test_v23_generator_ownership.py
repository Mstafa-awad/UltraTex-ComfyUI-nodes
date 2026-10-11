"""Pure offline verification of the v0.2.3 ownership fix.

Only ComfyUI node-type definitions are stubbed, not the math/behavior under test.
"""
from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import patch
from types import SimpleNamespace


def load_simple():
    if "comfy_api.latest" not in sys.modules:
        comfy_api=types.ModuleType("comfy_api")
        latest=types.ModuleType("comfy_api.latest")
        class Custom:
            def __init__(self,name): self.name=name
            def Input(self,*a,**k): return None
            def Output(self,*a,**k): return None
        io=SimpleNamespace(ComfyNode=type("ComfyNode",(),{}),Custom=Custom,
                           Mesh=SimpleNamespace(Input=lambda *x,**k:None,Output=lambda *x,**k:None),
                           Image=SimpleNamespace(Input=lambda *x,**k:None,Output=lambda *x,**k:None))
        latest.io=io
        comfy_api.latest=latest
        sys.modules["comfy_api"]=comfy_api
        sys.modules["comfy_api.latest"]=latest
    return importlib.import_module("comfyui_ultratex_modular.simple")


def test_resolution_and_count_are_owned_by_generation():
    simple=load_simple()
    source=simple.PreparedSource(SimpleNamespace(geometry_id="mesh-id"),object(),None,object(),"foreground","front")
    rendered=[]
    def fake_plan(ctx, render_size, sample_size, yaw):
        rendered.append((render_size,sample_size,yaw))
        return SimpleNamespace(identity=f"mesh-id:{render_size}:{sample_size}:{yaw}")
    def fake_gb(context,plan,policy): return SimpleNamespace(plan=plan),{}
    def fake_ref(*args): return "reference"
    with patch("comfyui_ultratex_modular.core.geometry.plan_views",fake_plan),\
         patch("comfyui_ultratex_modular.core.render.render_gbuffers",fake_gb),\
         patch("comfyui_ultratex_modular.core.conditions.prepare_reference",fake_ref):
        six=simple.setup_for_generation(source,1536,6)
        twelve=simple.setup_for_generation(source,2048,12)
    assert len(six.gbuffers)==1
    assert len(twelve.gbuffers)==2
    assert rendered==[(2048,1536,0.0),(2048,2048,0.0),(2048,2048,45.0)]


def test_generation_rejects_invalid_view_number():
    simple=load_simple()
    source=simple.PreparedSource(object(),object(),None,object(),"foreground","front")
    try:
        simple.setup_for_generation(source,1536,9)
    except ValueError as exc:
        assert "6 or 12" in str(exc)
    else:
        raise AssertionError("Unsupported count accepted")
