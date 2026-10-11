"""CPU tests added for the v0.2 shading/projection upgrade.
Run: python -m unittest comfyui_ultratex_modular.tests.test_quality_upgrade -v
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ..core.geometry import condition_normals
from ..core.bake import bilinear, repair_atlas
from ..core.guidance import reproject_atlas
from ..core.runtime import Store
from ..core.contracts import RuntimePolicy, AtlasTile, AtlasContext


class QualityUpgradeTests(unittest.TestCase):
    def test_faceted_normals_smooth_without_geometry_change(self):
        # Two UV-split triangles occupying one smoothly shaded planar square.
        v = np.array([[0,0,0],[1,0,0],[0,1,0],
                      [1,0,0],[1,1,0],[0,1,0]],np.float32)
        faces = np.array([[0,1,2],[3,4,5]],np.int32)
        authored=np.tile([0,0,1],(6,1)).astype(np.float32)
        authored[:3]=[.42,0,.907]
        authored[3:]=[-.42,0,.907]
        orig=v.copy()
        normal_auto=condition_normals(v,faces,authored,'auto_smooth')
        normal_source=condition_normals(v,faces,authored,'preserve')
        np.testing.assert_allclose(normal_auto, np.tile([0,0,1],(6,1)),atol=1e-6)
        self.assertGreater(np.abs(normal_source[:,0]).mean(),.3)
        np.testing.assert_array_equal(v,orig)

    def test_preserve_intentional_sharp_normals(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        authored=np.tile([1,0,0],(3,1)).astype(np.float32)
        n=condition_normals(v,np.array([[0,1,2]],np.int32),authored,'auto_smooth')
        np.testing.assert_allclose(n,authored)

    def test_bilinear_uv_reprojection_and_confidence(self):
        class RendererStub:
            def __init__(self,ctx): pass
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def draw(self,w,h,matrix,return_uv=False):
                assert return_uv
                out=np.zeros((h,w,4),np.float32)
                out[...,0]=.5
                out[...,1]=.5
                out[...,3]=1.
                return np.zeros_like(out),out
        ctx=object()
        plan=SimpleNamespace(sample_size=16,matrices=[np.eye(4)]*6)
        gbuffer=SimpleNamespace(plan=plan)
        tex=np.zeros((4,4,3),np.float32);tex[...,1]=1
        observed=np.ones((4,4),np.float32)
        confidence=np.ones((4,4),np.float32)
        with patch('comfyui_ultratex_modular.core.guidance.Renderer',RendererStub):
            images,keep=reproject_atlas(ctx,gbuffer,tex,observed,confidence)
        self.assertEqual(images.shape,(6,16,16,3))
        np.testing.assert_allclose(images[...,1],1.,atol=1e-6)
        np.testing.assert_allclose(keep,1.,atol=1e-6)

    def test_normal_aware_fill_rejects_opposing_surface(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(RuntimePolicy(temp,ram_cache_mb=16,system_headroom_mb=0))
            # First row: 2 valid texels on two nearly coincident opposing surfaces.
            # Second row: missing point on the upward face, closer to the DOWN
            # texel in Euclidean space but needing the UP texel color.
            pos=np.zeros((2,2,3),np.float32)
            pos[0,0]=[0,0,.01]  # upward, red
            pos[0,1]=[0,0,.001] # downward, blue
            pos[1,0]=[0,0,0]    # missing, upward
            nrm=np.zeros_like(pos)
            nrm[0,0]=[0,0,1]
            nrm[0,1]=[0,0,-1]
            nrm[1,0]=[0,0,1]
            chart=np.array([[1,1],[1,0]],np.int32)
            tile=AtlasTile(0,0,store.put(pos),store.put(nrm),store.put(np.array([[0,1],[2,-1]],np.int32)))
            atlas=SimpleNamespace(chart_ids=store.put(chart),tiles=[tile])
            tex=np.zeros((2,2,3),np.float32)
            tex[0,0]=[1,0,0];tex[0,1]=[0,0,1]
            holes=np.array([[0,0],[1,0]],np.float32)
            repaired,mask,_=repair_atlas(tex,holes,atlas,'normal_aware_surface',0)
            self.assertGreater(repaired[1,0,0],.99)
            self.assertLess(repaired[1,0,2],.01)
            np.testing.assert_allclose(repaired[0],tex[0])
            self.assertTrue(mask[1,0])


if __name__ == '__main__': unittest.main()

class BakeUpgradeTests(unittest.TestCase):
    def test_multiband_bake_keeps_a_constant_color_and_never_adds_triangles(self):
        from ..core.bake import bake_map
        with tempfile.TemporaryDirectory() as temp:
            policy=RuntimePolicy(temp,ram_cache_mb=16,system_headroom_mb=0)
            store=Store(policy)
            p=np.zeros((4,4,3),np.float32)
            normals=np.zeros_like(p);normals[...,2]=1.
            faces=np.ones((4,4),np.int32)
            tile=AtlasTile(0,0,store.put(p),store.put(normals),store.put(faces))
            plan=SimpleNamespace(geometry_id='mesh',identity='view',sample_size=16,matrices=[np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-1.67],[0,0,0,1]],np.float32)])
            depth=store.put(np.full((16,16),1.67,np.float32))
            mask=store.put(np.ones((16,16),bool))
            gb=SimpleNamespace(plan=plan,depths=(depth,))
            atlas=SimpleNamespace(geometry_id='mesh',gbuffer=gb,size=4,tiles=[tile],uv_coverage=store.put(np.ones((4,4),bool)),atlas_id='atlas')
            rgb=np.full((16,16,3),[200,50,10],np.uint8)
            views=SimpleNamespace(plan=plan,gbuffer=gb,task='albedo',images=(store.put(rgb),),masks=(mask,),job_id='fixture')
            baked,_,seen,_,_,report=bake_map(atlas,views,'base_color',policy,view_indices=(0,),
                                            edge_pixels=0,blend_mode='detail_preserving',gutter=0)
            assert np.isfinite(baked).all()
            np.testing.assert_allclose(baked,np.broadcast_to(rgb[0,0]/255,baked.shape),atol=.01)
            self.assertGreater(seen.mean(),.9)
            self.assertEqual(report['blend_mode'],'detail_preserving')
