"""Scoped OpenGL renderer: indexed geometry, compact view buffers, tiled UV rasterization."""
from __future__ import annotations

import numpy as np

from .contracts import GBuffer
from .geometry import FOCAL, GLTF_TO_BLENDER
from .runtime import Store, fingerprint, interrupt, measure, ram_preflight


class Renderer:
    VS = """#version 330
    uniform mat4 mv;
    uniform mat4 projection;
    uniform int uv_mode;
    uniform vec4 tile; // normalized left, bottom, width, height
    in vec3 in_pos;
    in vec3 in_nrm;
    in vec2 in_uv;
    out vec3 pos;
    out vec3 nrm;
    out float depth;
    out vec2 uv_coord;
    void main() {
        pos=in_pos; nrm=in_nrm; uv_coord=in_uv;
        vec4 p=mv*vec4(in_pos,1.);
        depth=-p.z;
        gl_Position=uv_mode==1 ? vec4(((in_uv-tile.xy)/tile.zw)*2.-1.,0.,1.) : projection*p;
    }"""
    FS = """#version 330
    in vec3 pos;
    in vec3 nrm;
    in float depth;
    in vec2 uv_coord;
    uniform int uv_mode;
    layout(location=0) out vec4 normal_depth;
    layout(location=1) out vec4 position_face;
    layout(location=2) out float count;
    void main() {
        float l=length(nrm);
        normal_depth=vec4(l>1e-10 ? nrm/l : vec3(0.),depth);
        position_face=uv_mode==2 ? vec4(uv_coord,0.,float(gl_PrimitiveID+1)) : vec4(pos,float(gl_PrimitiveID+1));
        count=1.;
    }"""

    def __init__(self, context):
        import moderngl
        self.ctx = None
        self.resources = []
        try:
            self.ctx = moderngl.create_standalone_context(require=330)
            self.prog = self.ctx.program(vertex_shader=self.VS, fragment_shader=self.FS)
            self.resources.append(self.prog)
            # Separate attribute buffers avoid a second complete interleaved geometry copy.
            arrays = [context.vertices, context.normals]
            uv = np.zeros((len(context.vertices), 2), np.float32) if context.uvs is None else np.array(context.uvs)
            if context.uv_convention == "v_down":
                uv[:, 1] = 1 - uv[:, 1]
            arrays.append(uv)
            buffers = [self.ctx.buffer(np.ascontiguousarray(a, dtype=np.float32)) for a in arrays]
            self.resources.extend(buffers)
            index = self.ctx.buffer(np.ascontiguousarray(context.faces, dtype=np.int32))
            self.resources.append(index)
            self.vao = self.ctx.vertex_array(self.prog, [(buffers[0], "3f", "in_pos"),
                                            (buffers[1], "3f", "in_nrm"), (buffers[2], "2f", "in_uv")], index)
            self.resources.append(self.vao)
        except Exception:
            self.close()
            raise

    def close(self):
        for resource in reversed(self.resources):
            resource.release()
        self.resources.clear()
        if self.ctx is not None:
            self.ctx.release()
            self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def draw(self, width, height, matrix, tile=None, count_only=False, return_uv=False):
        import moderngl
        ctx = self.ctx
        allocated = []
        try:
            a = ctx.texture((width, height), 4, dtype="f4")
            allocated.append(a)
            b = ctx.texture((width, height), 4, dtype="f4")
            allocated.append(b)
            c = ctx.texture((width, height), 1, dtype="f4")
            allocated.append(c)
            depth = ctx.depth_renderbuffer((width, height))
            allocated.append(depth)
            fbo = ctx.framebuffer([a, b, c], depth)
            allocated.append(fbo)
            fbo.use()
            ctx.viewport = (0, 0, width, height)
            fbo.clear(0., 0., 0., 0., depth=1.)
            ctx.disable(moderngl.CULL_FACE | moderngl.BLEND)
            if tile is None:
                ctx.enable(moderngl.DEPTH_TEST)
            else:
                ctx.disable(moderngl.DEPTH_TEST)
            if count_only:
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            near, far = .05, 10.
            proj = np.array([[2*FOCAL, 0, 0, 0], [0, 2*FOCAL, 0, 0],
                             [0, 0, (far+near)/(near-far), 2*far*near/(near-far)], [0,0,-1,0]], np.float32)
            self.prog["mv"].write(np.asarray(matrix, np.float32).T.copy())
            self.prog["projection"].write(proj.T.copy())
            self.prog["uv_mode"].value = 1 if tile is not None else (2 if return_uv else 0)
            self.prog["tile"].value = tile or (0., 0., 1., 1.)
            self.vao.render(moderngl.TRIANGLES)
            if count_only:
                return np.frombuffer(c.read(), np.float32).reshape(height, width)[::-1].copy()
            nd = np.frombuffer(a.read(), np.float32).reshape(height, width, 4)[::-1].copy()
            pf = np.frombuffer(b.read(), np.float32).reshape(height, width, 4)[::-1].copy() if (tile is not None or return_uv) else None
            return nd, pf
        finally:
            for resource in reversed(allocated):
                resource.release()


def render_gbuffers(context, plan, policy):
    if context.geometry_id != plan.geometry_id:
        raise ValueError("View plan belongs to different geometry")
    store = Store(policy)
    ns, ds, ms = [], [], []
    with measure("render_gbuffers", render_size=plan.render_size) as report:
        keys = [(fingerprint("normal-v2-auto", plan.identity, i), fingerprint("depth-v1", plan.identity, i),
                 fingerprint("mask-v1", plan.identity, i)) for i in range(6)]
        cached = [tuple(store.lookup(k) for k in keys_i) for keys_i in keys]
        if all(all(x is not None for x in triple) for triple in cached):
            return GBuffer(plan, tuple(x[0] for x in cached), tuple(x[1] for x in cached), tuple(x[2] for x in cached)), dict(report, cache_hit=True)
        ram_preflight(plan.render_size**2 * 32, policy.system_headroom_mb)
        with Renderer(context) as renderer:
            for i, matrix in enumerate(plan.matrices):
                interrupt()
                nd, _ = renderer.draw(plan.render_size, plan.render_size, matrix)
                mask = nd[..., 3] > 0
                n = nd[..., :3] @ np.asarray(plan.rotation, np.float32).T @ GLTF_TO_BLENDER.T
                n[..., 2] *= -1
                rgba = np.empty((*mask.shape, 4), np.uint8)
                rgba[..., :3] = np.clip((n * .5 + .5) * 255 + .5, 0, 255).astype(np.uint8)
                rgba[..., :3][~mask] = 0
                rgba[..., 3] = mask.astype(np.uint8) * 255
                ns.append(store.put(rgba, keys[i][0]))
                ds.append(store.put(nd[..., 3], keys[i][1]))
                ms.append(store.put(mask, keys[i][2]))
        report.update(cache_hit=False, stored_views=6, foreground_ratios=[float(x.read().mean()) for x in ms])
    return GBuffer(plan, tuple(ns), tuple(ds), tuple(ms)), report
