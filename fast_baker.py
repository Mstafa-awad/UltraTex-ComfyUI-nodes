"""Production baking node with retained modes and optional source-preserving fusion."""
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


# Retained modes preserve their calculations; sharp_source is a separate branch.
UPSCALE_MODES = ("fast_detail", "off_direct", "detail_consistent", "sharp_source", "sharp_coherent", "sharp_anchor", "sharp_anchor_v2")
SHARP_MODES = frozenset(("sharp_source", "sharp_coherent", "sharp_anchor", "sharp_anchor_v2"))


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
    if target not in (1024, 2048, 4096, 8192):
        raise ValueError("texture_size must be 1024, 2048, 4096 or 8192")
    if upscale_mode not in UPSCALE_MODES:
        raise ValueError(f"Unknown upscale_mode {upscale_mode!r}; choose one of {UPSCALE_MODES}")
    if target==8192 and upscale_mode not in SHARP_MODES:
        raise ValueError("8192 atlas requires a sharp mode. Use 4096 with the retained modes.")
    if completion not in ("surface_fill", "off"):
        raise ValueError("completion must be surface_fill or off")

    maps, coverage, report = bake_surface_maps(
        albedo_setup, albedo_views, target, completion == "surface_fill",
        mr_setup, mr_views, device_preference=device, detail=(upscale_mode in ("fast_detail", "detail_consistent")),
        consistent=(upscale_mode == "detail_consistent"),source_sharp=(upscale_mode == "sharp_source"),coherent=(upscale_mode=='sharp_coherent'),
        anchor=(upscale_mode in ('sharp_anchor','sharp_anchor_v2')),exact_support=(upscale_mode=='sharp_anchor_v2'))

    report["upscale_mode"] = upscale_mode
    report["output_size"] = target
    density = report.get("base_color", {}).get("harmonisation", {}).get("sampling_density")
    if density:
        report["sampling_density"] = dict(density, bake_size=target)

    context = albedo_setup.context
    t0 = time.perf_counter()
    infos = {name: MapInfo(context.atlas_id, name, target, target, provenance="utx-fast-surface-bake")
             for name in maps}
    tensors = {name: (torch.from_numpy(np.asarray(value,np.float32)[None]) if upscale_mode in SHARP_MODES else _image(value))
               for name, value in maps.items()}
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
        if upscale_mode in SHARP_MODES:
            # Keep the exact IMAGE shape/values without allocating two extra
            # 768MiB constant maps for an albedo-only 8K result.
            roughness=torch.full((1,1,1,3),r).expand_as(tensors['base_color'])
            metallic=torch.full((1,1,1,3),m).expand_as(tensors['base_color'])
        else:
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
            torch.from_numpy((np.asarray(coverage,np.float32) if upscale_mode in SHARP_MODES else np.array(coverage,np.float32))[None]), report_text)


class FastSurfaceTextureBaker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "UTXFastSurfaceTextureBaker", "3. Fast Surface Texture Baker",
            [
                SETUP.Input("albedo_setup"), VIEWS.Input("albedo_views"),
                io.Combo.Input("texture_size", options=[1024, 2048, 4096, 8192], default=2048,
                               tooltip="Final exported atlas size. UTX2.5 target changes source views only. "
                                       "8192 requires a sharp mode and takes more bake time and RAM."),
                io.Combo.Input("completion", options=["surface_fill", "off"], default="surface_fill",
                               tooltip="surface_fill: texels no camera saw are completed along the mesh surface "
                                       "(seamless across UV seams, never copied from unrelated UV islands). "
                                       "off: leave unseen Base Color black to inspect raw camera coverage "
                                       "(unseen MR becomes a neutral rough dielectric)."),
                io.Combo.Input("upscale_mode", options=list(UPSCALE_MODES), default="fast_detail",
                               tooltip="fast_detail: detail-preserving cubic reconstruction and best-camera "
                                       "fine detail, baked directly at texture_size. "
                                       "off_direct: original v0.4.0 direct bake. "
                                       "detail_consistent: experimental UV-correct filter footprints, full mips "
                                       "and narrower mid-band view blending; "
                                       "can reduce ghosting but may expose mismatched source views. "
                                       "sharp_source: preserve source-scale edges/markings after enhancement; "
                                       "blend only coarse colour, with bounded correction and no mesh-scale low-pass. "
                                       "Disagreeing views can still show transitions. "
                                       "sharp_anchor: accepted rc3 front-view RGB; other views cover hidden/grazing areas. "
                                       "sharp_anchor_v2: same anchor with exact mip/cubic support checks at surface boundaries. "
                                       "sharp_coherent: experimental automatic spatial patch selection with bounded log-colour seam correction. "
                                       "Adds a CPU graph-cut guide pass; retained sharp_source is the rollback. "
                                       "Crop transforms from UTX 2.5 work with all modes. No AI required."),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto",
                               tooltip="auto: attempt the configured ComfyUI CUDA device after model offload. "
                                       "No predicted-memory refusal or automatic CPU rerun; actual allocation errors propagate. "
                                       "cpu explicitly selects CPU execution."),
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

