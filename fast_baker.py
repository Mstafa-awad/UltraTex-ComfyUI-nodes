"""Single production baking node with two direct-resolution modes."""
from __future__ import annotations

import json
import logging
import time
import numpy as np
import torch
from comfy_api.latest import io

from .simple import SETUP, VIEWS, schema, validate_bake_branch, validate_shared_bake_geometry

log = logging.getLogger("UltraTex.FastBake")


def _image(array):
    return torch.from_numpy(np.array(array, np.float32)[None])


# Only two direct baking algorithms remain. Both bake at the requested resolution.
# Their implementation in core/fastbake.py is intentionally unchanged.
UPSCALE_MODES = ("fast_detail", "off_direct")


def run_fast_bake(albedo_setup, albedo_views, texture_size=2048, completion="surface_fill",
                  upscale_mode="fast_detail", device="auto", mr_setup=None, mr_views=None):
    from .core.contracts import MapInfo
    from .core.fastbake import bake_surface_maps
    from .core.materials import apply_materials, create_file

    start = time.perf_counter()
    validate_bake_branch(albedo_setup, albedo_views, "albedo")
    if (mr_setup is None) != (mr_views is None):
        raise ValueError("Connect BOTH mr_setup and mr_views (or neither for an Albedo-only bake)")
    if mr_setup is not None:
        validate_bake_branch(mr_setup, mr_views, "mr")
        validate_shared_bake_geometry(albedo_setup, mr_setup)
    target = int(texture_size)
    if target not in (1024, 2048, 4096):
        raise ValueError("texture_size must be 1024, 2048 or 4096")
    if upscale_mode not in UPSCALE_MODES:
        raise ValueError(f"Unknown upscale_mode {upscale_mode!r}; choose fast_detail or off_direct")
    if completion not in ("surface_fill", "off"):
        raise ValueError("completion must be surface_fill or off")

    maps, coverage, report = bake_surface_maps(
        albedo_setup, albedo_views, target, completion == "surface_fill",
        mr_setup, mr_views, device_preference=device, detail=(upscale_mode == "fast_detail"))

    report["upscale_mode"] = upscale_mode
    report["output_size"] = target
    density = report.get("base_color", {}).get("harmonisation", {}).get("sampling_density")
    if density:
        report["sampling_density"] = dict(density, bake_size=target)

    context = albedo_setup.context
    t0 = time.perf_counter()
    infos = {name: MapInfo(context.atlas_id, name, target, target, provenance="utx-fast-surface-bake")
             for name in maps}
    tensors = {name: _image(value) for name, value in maps.items()}
    mesh, _ = apply_materials(context.mesh, uv_convention=getattr(context, "uv_convention", "v_down"),
                              map_infos=infos, **tensors)
    glb = create_file(mesh)
    material = getattr(mesh, "material", None) or {}
    if "roughness" in tensors:
        roughness, metallic = tensors["roughness"], tensors["metallic"]
    else:
        r = material.get("roughness_factor", 0.7)
        m = material.get("metallic_factor", 0.0)
        r = 0.7 if r is None or r < 0 else float(r)
        m = 0.0 if m is None or m < 0 else float(m)
        roughness = torch.full_like(tensors["base_color"], r)
        metallic = torch.full_like(tensors["base_color"], m)
    report["material_export_seconds"] = round(time.perf_counter() - t0, 3)
    report["node_total_seconds"] = round(time.perf_counter() - start, 3)
    base = report.get("base_color", {})
    summary = (f"[UTX Fast Bake] {upscale_mode} {target}px on {report.get('device')} in "
               f"{report['bake_total_seconds']:.2f}s bake / "
               f"{report['node_total_seconds']:.2f}s node; observed {100 * base.get('observed_area_fraction', 0):.1f}% "
               f"of surface area, {100 * base.get('unobserved_fraction', 0):.1f}% of texels completed")
    if density and density.get("view_px_per_texel_median"):
        summary += (f"; views sample the {target}px atlas at {density['view_px_per_texel_median']:.2f} "
                    f"view px per texel")
    log.info(summary)
    report_text = json.dumps(report, indent=2, default=str)
    return (mesh, glb, tensors["base_color"], roughness, metallic,
            torch.from_numpy(np.array(coverage, np.float32)[None]), report_text)


class FastSurfaceTextureBaker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "UTXFastSurfaceTextureBaker", "3. Fast Surface Texture Baker",
            [
                SETUP.Input("albedo_setup"), VIEWS.Input("albedo_views"),
                io.Combo.Input("texture_size", options=[1024, 2048, 4096], default=2048),
                io.Combo.Input("completion", options=["surface_fill", "off"], default="surface_fill",
                               tooltip="surface_fill: texels no camera saw are completed along the mesh surface "
                                       "(seamless across UV seams, never copied from unrelated UV islands). "
                                       "off: leave unseen Base Color black to inspect raw camera coverage "
                                       "(unseen MR becomes a neutral rough dielectric)."),
                io.Combo.Input("upscale_mode", options=list(UPSCALE_MODES), default="fast_detail",
                               tooltip="fast_detail: detail-preserving cubic reconstruction and best-camera "
                                       "fine detail, baked directly at texture_size. "
                                       "off_direct: original v0.4.0 direct bake. "
                                       "Neither mode upscales images or requires additional models."),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto",
                               tooltip="auto: use the ComfyUI CUDA device (ComfyUI offloads models only if VRAM is "
                                       "short); falls back to CPU when there is still no room or CUDA runs out of memory."),
                SETUP.Input("mr_setup", optional=True), VIEWS.Input("mr_views", optional=True),
            ],
            [
                io.Mesh.Output(display_name="mesh"),
                io.File3DGLB.Output(display_name="glb"),
                io.Image.Output(display_name="base_color"),
                io.Image.Output(display_name="roughness"),
                io.Image.Output(display_name="metallic"),
                io.Mask.Output(display_name="observed_coverage"),
                io.String.Output(display_name="report"),
            ],
            "Production baker: plane-consistent depth visibility against the exact generation G-buffers, smooth "
            "confidence-weighted fusion with per-view colour matching, and surface-aware completion of texels no "
            "camera could see (never left black). fast_detail keeps the finest Base Color detail sharp (cubic "
            "reconstruction, fine band from the best camera). No forensic instrumentation, no OpenGL needed.",
        )

    @classmethod
    def execute(cls, albedo_setup, albedo_views, texture_size, completion="surface_fill", upscale_mode="fast_detail",
                device="auto", mr_setup=None, mr_views=None):
        return io.NodeOutput(*run_fast_bake(albedo_setup, albedo_views, int(texture_size), completion,
                                            upscale_mode, device, mr_setup, mr_views))


FAST_BAKER_NODES = [FastSurfaceTextureBaker]
