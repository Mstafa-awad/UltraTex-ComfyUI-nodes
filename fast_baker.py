"""Two supported bake presets with migration of retired workflow values."""
from __future__ import annotations

import json
import logging
import time
import numpy as np
import torch
from comfy_api.latest import io

from .simple import SETUP, VIEWS, schema, validate_bake_branch, validate_shared_bake_geometry

log = logging.getLogger("UltraTex.FastBake")


EDGE_REPAIR = "Quality + edge repair"
UPSCALE_MODES = ("Fast", "Quality", EDGE_REPAIR)
LEGACY_PRESETS = dict(fast_detail="Fast",off_direct="Fast",detail_consistent="Quality",sharp_source="Quality",
                      sharp_coherent="Quality",sharp_anchor="Quality",sharp_anchor_v2="Quality",sharp_anchor_v3=EDGE_REPAIR,sharp_anchor_v4=EDGE_REPAIR)

def normalize_preset(value):
    if value in UPSCALE_MODES:return value
    if value in LEGACY_PRESETS:return LEGACY_PRESETS[value]
    if isinstance(value,str) and value.lower() in ('fast','quality'):return value.capitalize()
    raise ValueError(f"Unknown bake quality {value!r}; choose one of {', '.join(UPSCALE_MODES)}")


def run_fast_bake(albedo_setup, albedo_views, texture_size=2048, completion="surface_fill",
                  upscale_mode="Quality", device="auto", mr_setup=None, mr_views=None):
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
    requested=upscale_mode;upscale_mode=normalize_preset(upscale_mode)
    if completion not in ("surface_fill", "off"):
        raise ValueError("completion must be surface_fill or off")

    maps, coverage, report = bake_surface_maps(
        albedo_setup, albedo_views, target, completion == "surface_fill",
        mr_setup, mr_views, device_preference=device,anchor=True,anchor_fast=upscale_mode=='Fast',
        exact_support=upscale_mode!='Fast',edge_margin=8 if upscale_mode==EDGE_REPAIR else 0)

    report["upscale_mode"] = upscale_mode
    if requested!=upscale_mode:report['migrated_preset']=dict(previous=requested,current=upscale_mode)
    report["output_size"] = target
    density = report.get("base_color", {}).get("harmonisation", {}).get("sampling_density")
    if density:
        report["sampling_density"] = dict(density, bake_size=target)

    context = albedo_setup.context
    t0 = time.perf_counter()
    infos = {name: MapInfo(context.atlas_id, name, target, target, provenance="utx-fast-surface-bake")
             for name in maps}
    tensors = {name:torch.from_numpy(np.asarray(value,np.float32)[None]) for name,value in maps.items()}
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
        # Scalar material outputs share storage at every atlas size.
        roughness=torch.full((1,1,1,3),r).expand_as(tensors['base_color'])
        metallic=torch.full((1,1,1,3),m).expand_as(tensors['base_color'])
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
            torch.from_numpy(np.asarray(coverage,np.float32)[None]), report_text)


class FastSurfaceTextureBaker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "UTXFastSurfaceTextureBaker", "3. Fast Surface Texture Baker",
            [
                SETUP.Input("albedo_setup"), VIEWS.Input("albedo_views"),
                io.Combo.Input("texture_size", options=[1024, 2048, 4096, 8192], default=2048,
                               tooltip="Final texture size. Larger textures take more bake time and memory."),
                io.Combo.Input("upscale_mode",display_name="bake_quality",options=list(UPSCALE_MODES),default="Quality",
                               tooltip="Fast: quicker projection for previews. "
                                       "Quality: exact texel support, keeps thin details (the 2.2 sharp_anchor_v2 bake plus the "
                                       "2.3 camera fixes). Quality + edge repair: also rejects source pixels near geometric "
                                       "edges and recolours them; removes some silhouette colour fringes but blurs thin parts "
                                       "(hat bands, braids) and is slower. All complete unseen areas and preserve the mesh and UVs."),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto",advanced=True,
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
            "Bake the generated views onto the original mesh. Choose Fast for iteration or Quality for final export. "
            "Texture resolution and camera fitting are independent of this choice.",
        )

    @classmethod
    def validate_inputs(cls,upscale_mode):
        try:normalize_preset(upscale_mode)
        except ValueError as error:return str(error)
        return True

    @classmethod
    def execute(cls, albedo_setup, albedo_views, texture_size, completion="surface_fill", upscale_mode="Quality",
                device="auto", mr_setup=None, mr_views=None):
        return io.NodeOutput(*run_fast_bake(albedo_setup, albedo_views, int(texture_size), completion,
                                            upscale_mode, device, mr_setup, mr_views))


FAST_BAKER_NODES = [FastSurfaceTextureBaker]

