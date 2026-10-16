"""Three main stages with sensible defaults; advanced nodes remain compatible.

Patched build:
- fast modular generation path retained
- standalone UV unwrap node using the official UltraTex unwrap method
- preview outputs for normals/reference/mask
- slider-based resolutions
- exposed guidance/denoise/LoRA strength sampling controls
- Bake node also outputs roughness and metallic maps as IMAGEs
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io

from .core.contracts import RuntimePolicy, TextureSetup
from .core.diagnostics import diagnose_node, monitor_stage as _diag_stage, active as _active_diag

SETUP = io.Custom("UTX_TEXTURE_SETUP")
SOURCE = io.Custom("UTX_PREPARED_SOURCE_V23")
VIEWS = io.Custom("UTX_VIEW_COLLECTION")
POLICY = io.Custom("UTX_RUNTIME_POLICY")


def schema(name, label, inputs, outputs, description):
    return io.Schema(
        node_id=name,
        display_name="UTX " + label,
        category="UltraTex Modular",
        inputs=inputs,
        outputs=outputs,
        description=description,
    )


def _mesh_from_arrays(source_mesh, vertices, faces, uvs=None, normals=None):
    """Create a native ComfyUI-style MESH object by patching a copy of source_mesh."""
    result = copy.copy(source_mesh)
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    result.vertices = torch.from_numpy(vertices.copy())[None]
    result.faces = torch.from_numpy(faces.copy())[None]
    result.vertex_counts = torch.tensor([len(vertices)], dtype=torch.int32)
    result.face_counts = torch.tensor([len(faces)], dtype=torch.int32)
    if normals is not None:
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        result.normals = torch.from_numpy(normals.copy())[None]
    if uvs is not None:
        uvs = np.ascontiguousarray(uvs, dtype=np.float32)
        result.uvs = torch.from_numpy(uvs.copy())[None]
    return result


def _unwrap_mesh(mesh, uv_mode="keep_existing", atlas_size=4096, padding=4):
    """Official UltraTex UV unwrap path.

    comfy_gpu uses ComfyUI's GPU unwrapper exactly like the official UltraTex node.
    xatlas is the CPU fallback. keep_existing preserves authored UVs.
    """
    from .core.geometry import mesh_arrays

    vertices, faces, uv, normals = mesh_arrays(mesh)
    if uv_mode == "keep_existing" and uv is not None and len(uv) == len(vertices):
        return mesh
    if uv_mode == "keep_existing":
        uv_mode = "comfy_gpu"

    if uv_mode == "comfy_gpu":
        try:
            import comfy.model_management as mm
            from comfy_extras import nodes_mesh_postprocess as mp

            device = mm.get_torch_device()
            mp._prepare_gpu_mesh_processing(device, len(faces) * 14 * 1024)
            vmapping, new_faces, new_uvs = mp._uv_unwrap(
                torch.from_numpy(np.ascontiguousarray(vertices, dtype=np.float32)).to(device),
                torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64)).to(device),
                "pec", int(atlas_size), int(padding), 0.0,
            )
            # ComfyUI unwrap returns native MESH UVs (v-down).
            # The official UltraTex wrapper flips them to OpenGL v-up because it
            # produces a trimesh. Do NOT flip here: our output is native MESH,
            # and the modular renderer performs that conversion internally.
            vmapping = vmapping.detach().cpu().numpy() if isinstance(vmapping, torch.Tensor) else np.asarray(vmapping)
            new_faces = new_faces.detach().cpu().numpy() if isinstance(new_faces, torch.Tensor) else np.asarray(new_faces)
            new_uvs = new_uvs.detach().cpu().numpy() if isinstance(new_uvs, torch.Tensor) else new_uvs
            vmapping = np.asarray(vmapping, dtype=np.int64)
            new_faces = np.asarray(new_faces, dtype=np.int32)
            new_uvs = np.array(new_uvs, dtype=np.float32, copy=True)
        except Exception as exc:
            raise RuntimeError(f"UltraTex comfy_gpu UV unwrap failed: {exc}. Try xatlas or a mesh with existing UVs.")
    elif uv_mode == "xatlas":
        try:
            import xatlas
        except Exception as exc:
            raise RuntimeError(f"UltraTex xatlas unwrap requested but xatlas is unavailable: {exc}")
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
        pack = xatlas.PackOptions()
        pack.resolution = int(atlas_size)
        pack.padding = int(padding)
        pack.bilinear = True
        atlas.generate(xatlas.ChartOptions(), pack)
        vmapping, new_faces, new_uvs = atlas[0]
        vmapping = np.asarray(vmapping, dtype=np.int64)
        new_faces = np.asarray(new_faces, dtype=np.int32)
        # xatlas uses OpenGL v-up, so convert it to native ComfyUI v-down.
        new_uvs = np.array(new_uvs, dtype=np.float32, copy=True)
        new_uvs[:, 1] = 1.0 - new_uvs[:, 1]
    else:
        raise ValueError("uv_mode must be keep_existing, comfy_gpu, or xatlas")

    new_vertices = vertices[vmapping]
    new_normals = normals[vmapping] if normals is not None else None
    return _mesh_from_arrays(mesh, new_vertices, new_faces, new_uvs, new_normals)


class UnwrapMeshUV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "UTXUnwrapMeshUVV25",
            "0. UV Unwrap Mesh",
            [
                io.Mesh.Input("mesh"),
                io.Int.Input("atlas_size", default=4096, min=512, max=8192, step=16),
                io.Combo.Input("uv_mode", options=["comfy_gpu", "xatlas", "keep_existing"], default="comfy_gpu"),
                io.Int.Input("padding", default=4, min=0, max=64, step=1),
            ],
            [io.Mesh.Output(display_name="uv_mesh")],
            "Standalone UV node. Uses the official UltraTex comfy_gpu / xatlas unwrap path and outputs a UV-ready native MESH.",
        )

    @classmethod
    def execute(cls, mesh, atlas_size, uv_mode, padding):
        return io.NodeOutput(_unwrap_mesh(mesh, uv_mode, int(atlas_size), int(padding)))


def _preview_normals(gbuffer, preview_size=512):
    from PIL import Image

    images = []
    for ref in gbuffer.normals:
        rgba = np.asarray(Image.fromarray(ref.read()).resize((preview_size, preview_size), Image.Resampling.BILINEAR))
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32) / 255.0 * alpha[..., None] + 1 - alpha[..., None]
        images.append(torch.from_numpy(rgb[None]))
    return torch.cat(images, dim=0)


def _preview_reference(reference, preview_size=512):
    from PIL import Image

    rgba = np.asarray(Image.fromarray(reference.rgba.read()).resize((preview_size, preview_size), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0
    alpha = rgba[..., 3]
    rgb = rgba[..., :3] * alpha[..., None] + 1 - alpha[..., None]
    return torch.from_numpy(rgb[None]), torch.from_numpy(alpha[None])


@dataclass(frozen=True)
class PreparedSource:
    """Prepared mesh + original reference, without baked-in view resolution/count.

    The Generate node owns its camera/sample settings. Keeping the *original*
    image/mask avoids scaling the 512 px preview up to production resolution.
    """
    context: object
    reference_image: object
    reference_mask: object
    runtime: RuntimePolicy
    mask_polarity: str
    alignment: str
    # UTX 1-owned, DISK-backed normal/depth conditioning and camera controls.
    # All defaults preserve backward compatibility with old PreparedSource calls.
    normal_gbuffer: object = None
    normal_resolution: int = 1024
    camera_fit: str = "native_mesh_fit"
    image_margin_percent: float = 2.0
    image_padding_pixels: int = 8
    reference_zoom: float = 1.0

    def _comfy_cache_tensors(self):
        return self.context._comfy_cache_tensors()


def prepare_source(mesh, image, mask=None, alignment="front", mask_polarity="background",
                   runtime=None, normal_mode="auto_smooth"):
    import folder_paths
    from .core.geometry import prepare_geometry

    runtime = runtime or RuntimePolicy(str(Path(folder_paths.get_output_directory()) / '.ultratex_cache'))
    prepared_mesh = mesh
    context, _ = prepare_geometry(prepared_mesh, policy=runtime, normal_mode=normal_mode)
    if context.uvs is None:
        raise ValueError("Mesh has no UVs. Use the standalone 'UTX 0. UV Unwrap Mesh' node first, or provide a mesh with clean UVs.")
    # Keep UV-split geometry and corrected normals together for export and bake.
    nv = len(context.vertices)
    nf = len(context.faces)
    prepared = _mesh_from_arrays(
        prepared_mesh,
        prepared_mesh.vertices[0, :nv].detach().cpu().numpy(),
        prepared_mesh.faces[0, :nf].detach().cpu().numpy(),
        context.uvs,
        context.normals,
    )
    context = replace(context, mesh=prepared)
    if mask_polarity == "background (LoadImage)":
        mask_polarity = "background"
    elif mask_polarity == "foreground (BiRefNet)":
        mask_polarity = "foreground"
    return PreparedSource(context, image, mask, runtime, mask_polarity, alignment)


def _utx1_auto_fit_camera_distance(context, base_matrices, target_fill=0.94):
    """Fit the WHOLE used mesh inside all six cameras with a small safe border.

    Our renderer and baker share a fixed calibrated FOCAL. Therefore zooming by
    cropping or changing FOCAL alone would break correspondence. Move the cameras
    *along their own forward axis* by a SINGLE distance adjustment instead.
    The exact updated 4x4 matrices are then used by generation, depth and bake.

    target_fill is the fraction of image width/height available to the mesh.
    A single distance for the six-camera rig prevents varying scale between views.
    """
    from .core.geometry import FOCAL
    target_fill = float(target_fill)
    if not np.isfinite(target_fill) or not 0.5 <= target_fill <= 0.98:
        raise ValueError("target_fill must be in [0.5, 0.98]")
    # Large meshes have many repeated UV-split vertices. For fitting, the
    # bounding geometry is cheap to scan and conservative even for loose verts.
    vertices = np.asarray(context.vertices, np.float32)
    threshold = target_fill * 0.5
    needed_shift = -float("inf")
    # Avoid allocating N*6 arrays on high-poly TRELLIS meshes.
    for matrix in base_matrices:
        m = np.asarray(matrix, np.float32)
        rotation = m[:3, :3]
        translation = m[:3, 3]
        for begin in range(0, len(vertices), 65536):
            block = vertices[begin:begin + 65536] @ rotation.T + translation
            depth = -block[:, 2]
            desired_depth = (FOCAL / threshold) * np.maximum(np.abs(block[:, 0]), np.abs(block[:, 1]))
            # Future depth = original depth + shift. Minimum positive clearance
            # is also enforced for vertices on the centre ray.
            needed_shift = max(needed_shift, float(np.max(np.maximum(desired_depth, 0.1) - depth)))
    if not np.isfinite(needed_shift):
        raise ValueError("Cannot determine a finite camera fit")
    # Fit can move inward or outward, but prevent extreme zooming for unusually
    # thin/flat meshes (the 9B adapter is calibrated to a canonical camera rig).
    return max(needed_shift, -0.40)  # at most 0.40 normalized scene units closer



def _utx1_plan_views(context, render_size=2048, sample_size=1024, yaw=0., elevation=0., framing="original"):
    """UTX 1's v0.4.2 camera-framing plan on the untouched v0.4.1 geometry core."""
    from .core.geometry import plan_views
    from .core.runtime import fingerprint

    # v0.4.1's original camera rotations/translations are identical to v0.4.2.
    base = plan_views(context, render_size, sample_size, yaw, elevation)
    if framing not in ("original", "auto_fit_94"):
        raise ValueError("framing must be original or auto_fit_94")
    matrices = [np.array(m, dtype=np.float32) for m in base.matrices]
    if framing == "auto_fit_94":
        shift = _utx1_auto_fit_camera_distance(context, matrices)
        for matrix in matrices:
            matrix[2, 3] -= shift
    matrices = tuple(tuple(tuple(float(v) for v in row) for row in m) for m in matrices)
    return replace(base, matrices=matrices, identity=fingerprint(
        "views-v2-framing", context.geometry_id, matrices, render_size, sample_size))


def _utx1_calibrated_plan(context, render_size, sample_size, yaw, camera_fit, margin_percent, padding_pixels):
    """UTX 1 camera fitting ONLY. No image crop, no GPU tensor and no geometry changes.

    Native uses v0.4.2's EXACT calibrated auto_fit_94 distance algorithm.
    Custom changes only the target margin, keeping its <=0.40 inward safety cap
    and *single shared* fit for all six cameras (no new aggressive per-view zoom).
    """
    plan_views = _utx1_plan_views
    _auto_fit_camera_distance = _utx1_auto_fit_camera_distance
    from .core.runtime import fingerprint

    size = int(render_size)
    mode = str(camera_fit)
    if mode == "original":
        return plan_views(context, size, int(sample_size), float(yaw), framing="original")
    if mode == "native_mesh_fit":
        return plan_views(context, size, int(sample_size), float(yaw), framing="auto_fit_94")
    if mode != "custom":
        raise ValueError("camera_fit must be original, native_mesh_fit, or custom")
    margin = float(margin_percent)
    padding = int(padding_pixels)
    if not 0 <= margin <= 20 or padding < 0 or padding > 256:
        raise ValueError("Custom camera image margin must be 0..20%, padding 0..256 pixels")
    # The independent margin controls act on camera placement, NOT image resizing.
    fill = 1.0 - 2.0*(margin/100.0 + padding/size)
    fill = min(0.98, max(0.50, fill))
    plan = plan_views(context, size, int(sample_size), float(yaw), framing="original")
    matrices = [np.array(m, dtype=np.float32) for m in plan.matrices]
    shift = _auto_fit_camera_distance(context, matrices, target_fill=fill)
    for matrix in matrices:
        matrix[2, 3] -= shift
    matrices = tuple(tuple(tuple(float(v) for v in row) for row in m) for m in matrices)
    return replace(plan, matrices=matrices,
                   identity=fingerprint("utx1-v042-custom-camera-v1", plan.identity, matrices, margin, padding))


def _utx1_zoom_aligned_ref(reference, zoom, policy):
    """Optional UTX 1 reference zoom: IMAGE+alpha/mask together; no normal-view crop."""
    if abs(float(zoom)-1.0) < 1e-6:
        return reference
    from PIL import Image
    from .core.runtime import Store, fingerprint
    z=float(zoom)
    if not 0.5 <= z <= 1.5:
        raise ValueError("reference_zoom must be between 0.5 and 1.5")
    src = Image.fromarray(np.array(reference.rgba.read(), copy=True), mode="RGBA")
    w,h=src.size
    nw,nh=max(1,round(w*z)),max(1,round(h*z))
    scaled=src.resize((nw,nh),Image.Resampling.LANCZOS)
    canvas=Image.new("RGBA",(w,h),(0,0,0,0))
    canvas.paste(scaled,((w-nw)//2,(h-nh)//2))
    arr=np.asarray(canvas)
    identity=fingerprint("utx1-reference-zoom-v1",reference.identity,z,w,h)
    rgba=Store(policy).put(arr,identity)
    return replace(reference,rgba=rgba,identity=identity,transform={**reference.transform,"utx1_zoom":z})


def _four_diagonal_views(gbuffer, policy):
    """Use 4 side-facing 45-degree cameras within UltraTex's 6-tile DiT grid.

    Top/bottom 45-degree cameras mostly duplicate the primary top/bottom.  We
    preserve the spatial grid and camera indexing required by the existing
    LoRA while making the two unused tiles truly inactive (zero foreground).
    """
    from .core.contracts import GBuffer
    from .core.runtime import Store, fingerprint

    active=(0, 1, 2, 3)
    plan=replace(gbuffer.plan, active_indices=active,
                 identity=fingerprint("utx-10v-four-diagonal-v1", gbuffer.plan.identity, active))
    empty=Store(policy).put(np.zeros((plan.render_size, plan.render_size), dtype=np.bool_))
    return GBuffer(plan, gbuffer.normals, gbuffer.depths,
                   gbuffer.masks[:4]+(empty, empty))


def setup_for_generation(source, resolution, view_count, framing="auto_fit_94"):
    """v0.4.2 generator setup; ONLY UTX 1's cached fitted rig is bridged in.

    UTX 2 NODE CLASS and the complete sampler remain BYTE-FOR-BYTE v0.4.1.
    Only a source prepared by UTX 1 activates this camera/G-buffer path.
    """
    from .core.geometry import plan_views
    from .core.render import render_gbuffers
    from .core.conditions import prepare_reference
    from .core.runtime import fingerprint

    resolution = int(resolution)
    view_count = int(view_count)
    if resolution < 256 or resolution > 4096 or resolution % 16:
        raise ValueError("Generate resolution must be a multiple of 16 between 256 and 4096")
    if view_count not in (6, 10, 12):
        raise ValueError("UltraTex supports calibrated 6, 10 or 12 views")
    prepared = getattr(source, "normal_gbuffer", None)
    use_utx1 = prepared is not None and framing == "auto_fit_94"
    if not use_utx1:
        # Unmodified v0.4.1 camera plan when no v0.4.2 UTX 1 conditioning is supplied.
        gbuffers=[]
        for yaw in ((0.0,) if view_count == 6 else (0.0, 45.0)):
            plan = plan_views(source.context, max(2048, resolution), resolution, yaw)
            rendered=render_gbuffers(source.context, plan, source.runtime)[0]
            if len(gbuffers)==1 and view_count==10:
                rendered=_four_diagonal_views(rendered,source.runtime)
            gbuffers.append(rendered)
        reference = prepare_reference(source.reference_image, source.reference_mask,
                                      gbuffers[0], source.runtime,
                                      source.mask_polarity, source.alignment)
        return TextureSetup(source.context, tuple(gbuffers), reference, source.runtime)

    from .core.contracts import GBuffer
    actual = int(getattr(source,"normal_resolution",prepared.plan.render_size))
    # UTX 1 owns the cameras and native normals; use the exact stored buffers
    # when their resolution suffices. If RGB output is larger, rerender on same
    # camera rig at full native size; never upscale conditioning images.
    if actual >= resolution:
        plan=replace(prepared.plan,sample_size=resolution,
                     identity=fingerprint("utx1-sample-v1",prepared.plan.identity,resolution))
        first=GBuffer(plan,prepared.normals,prepared.depths,prepared.masks)
    else:
        render_size=max(2048,resolution)
        plan=_utx1_calibrated_plan(source.context,render_size,resolution,0.0,
                                  source.camera_fit,source.image_margin_percent,source.image_padding_pixels)
        first=render_gbuffers(source.context,plan,source.runtime)[0]
    gbuffers=[first]
    if view_count in (10, 12):
        render_size=max(actual,resolution)
        plan=_utx1_calibrated_plan(source.context,render_size,resolution,45.0,
                                  source.camera_fit,source.image_margin_percent,source.image_padding_pixels)
        secondary=render_gbuffers(source.context,plan,source.runtime)[0]
        if view_count==10:
            secondary=_four_diagonal_views(secondary,source.runtime)
        gbuffers.append(secondary)
    reference=prepare_reference(source.reference_image,source.reference_mask,
                                first,source.runtime,source.mask_polarity,source.alignment)
    reference=_utx1_zoom_aligned_ref(reference,source.reference_zoom,source.runtime)
    return TextureSetup(source.context,tuple(gbuffers),reference,source.runtime)


class PrepareTexture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "UTXPrepareTextureV25", "1. Prepare Mesh + Reference", [
                io.Mesh.Input("mesh"), io.Image.Input("reference"),
                io.Combo.Input("alignment", options=["front", "center"], default="front"),
                io.Combo.Input("mask_polarity", options=["background", "foreground"], default="foreground",
                    tooltip="BiRefNet foreground mask: foreground. Load Image MASK: background."),
                io.Combo.Input("normal_mode", options=["auto_smooth", "preserve", "smooth"], default="auto_smooth"),
                io.Combo.Input("normal_view_resolution", options=[1024, 2048, 4096], default=1024,
                    tooltip="Actual mesh normal/depth render size. Images are disk-cached; only a 512 preview goes to UI."),
                io.Combo.Input("camera_fit", options=["native_mesh_fit", "original", "custom"], default="native_mesh_fit",
                    tooltip="native_mesh_fit = ORIGINAL v0.4.2 auto_fit_94 cameras (safe shared fit); custom adjusts camera margins; NO image crop."),
                io.Float.Input("image_margin_percent", default=2.0, min=0.0, max=20.0, step=0.25,
                    tooltip="Custom camera fit only: percent border EACH image side; not UV padding."),
                io.Int.Input("image_padding_pixels", default=8, min=0, max=256, step=1,
                    tooltip="Custom camera fit only: extra pixel safety border EACH image side; not UV padding."),
                io.Float.Input("reference_zoom", default=1.0, min=0.5, max=1.5, step=0.05,
                    tooltip="Optional aligned reference+mask zoom together (default 1.0 unchanged)."),
                io.Mask.Input("mask", optional=True), POLICY.Input("runtime", optional=True),
            ], [
                SOURCE.Output(display_name="prepared_source"),
                io.Mesh.Output(display_name="prepared_mesh"),
                io.Image.Output(display_name="normal_views_512"),
                io.Image.Output(display_name="aligned_reference_512"),
                io.Mask.Output(display_name="aligned_mask_512"),
            ], "Mesh normals/depth, reference and camera fitting ONLY. UTX 2 sampling/VRAM code remains v0.4.2."
        )

    @classmethod
    def execute(cls, mesh, reference, alignment, mask_polarity,
                normal_mode="auto_smooth", normal_view_resolution=1024,
                camera_fit="native_mesh_fit", image_margin_percent=2.0,
                image_padding_pixels=8, reference_zoom=1.0, mask=None, runtime=None):
        from .core.render import render_gbuffers
        from .core.conditions import prepare_reference
        from .core.contracts import GBuffer
        source = prepare_source(mesh, reference, mask, alignment, mask_polarity, runtime, normal_mode)
        size=int(normal_view_resolution)
        if size not in (1024,2048,4096):
            raise ValueError("normal_view_resolution must be 1024, 2048 or 4096")
        plan=_utx1_calibrated_plan(source.context,size,size,0.0,camera_fit,
                                   image_margin_percent,image_padding_pixels)
        gb,_=render_gbuffers(source.context,plan,source.runtime)
        # Disk-backed buffers: no new resident model or GPU activations.
        source=replace(source, normal_gbuffer=gb, normal_resolution=size,
                       camera_fit=camera_fit,image_margin_percent=float(image_margin_percent),
                       image_padding_pixels=int(image_padding_pixels),reference_zoom=float(reference_zoom))
        print(f"[UTX 1] Camera fit={camera_fit}, normals={size}px, "
              "RGB/Depth/Mask stored; UTX 2 retains original v0.4.2 sampler.")
        # Preview is only a downsample of already-rendered REAL mesh normals.
        preview_plan=replace(plan,sample_size=512)
        preview_gb=GBuffer(preview_plan,gb.normals,gb.depths,gb.masks)
        ref=prepare_reference(source.reference_image,source.reference_mask,
                              preview_gb,source.runtime,source.mask_polarity,source.alignment)
        ref=_utx1_zoom_aligned_ref(ref,source.reference_zoom,source.runtime)
        return io.NodeOutput(source,source.context.mesh,_preview_normals(gb,512),
                             *_preview_reference(ref,512))


class GenerateTextureViews(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths
        from .extension import decoder_names
        return schema(
            "UTXGenerateTextureViewsV23",
            "2. Generate Views + LoRA",
            [
                SOURCE.Input("source"), io.Model.Input("model"), io.Vae.Input("vae"), io.Conditioning.Input("positive"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Combo.Input("decoder_name", options=decoder_names()),
                io.Combo.Input("task", options=["albedo", "mr"], default="albedo"),
                io.Int.Input("view_resolution", default=1024, min=256, max=4096, step=16,
                    tooltip="Must match the resolution chosen in Prepare."),
                io.Combo.Input("view_count", options=[6, 10, 12], default=6,
                    tooltip="Must match the view count chosen in Prepare."),
                io.Int.Input("seed", default=0, min=0, max=2**63 - 1, control_after_generate=True),
                io.Int.Input("steps", default=4, min=1, max=100),
                io.Float.Input("guidance", default=4.0, min=0.0, max=20.0, step=0.1,
                    tooltip="CFG scale: active only when a different negative CONDITIONING is connected. With no negative input, it does not change sampling."),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("sparse_retention", default=0.20, min=0.01, max=1.0, step=0.01,
                    tooltip="Lower is lighter/faster; higher keeps more tokens in sparse attention."),
                io.Combo.Input("extra_angle_mode", options=["guided", "independent"], default="guided",
                    tooltip="For 10 or 12 views, use the first pass to guide the 45-degree pass. 10 views omits redundant rotated top/bottom."),
                io.Float.Input("extra_denoise", default=0.55, min=0.1, max=1.0, step=0.05,
                    tooltip="How freely the guided 45-degree pass can change already observed details."),
                io.Int.Input("extra_steps", default=16, min=1, max=100,
                    tooltip="Separate iteration budget for the 45-degree pass."),
                io.Boolean.Input("tiled_decode", default=True,
                    tooltip="Tiled VAE for limited VRAM. Disable only if the VAE can decode without OOM."),
                io.Conditioning.Input("negative", optional=True),
            ],
            [VIEWS.Output(), io.Image.Output(display_name="all_views_512"), io.Image.Output(display_name="contact_sheet"),
             SETUP.Output(display_name="generation_setup_to_bake")],
            "Fast generation node with exposed LoRA strength, CFG guidance, denoise, and sparse retention.",
        )

    @classmethod
    def validate_inputs(cls, lora_name, task, **kwargs):
        from .extension import ApplyTextureLora
        return ApplyTextureLora.validate_inputs(lora_name, task)

    @classmethod
    def fingerprint_inputs(cls, lora_name, decoder_name, **kwargs):
        import folder_paths
        from .extension import decoder_names
        from .core.models import file_hash
        decoder_names()
        return (
            file_hash(folder_paths.get_full_path_or_raise('loras', lora_name)),
            file_hash(folder_paths.get_full_path_or_raise('ultratex', decoder_name)),
        )

    @classmethod
    @diagnose_node
    def execute(
        cls,
        source,
        model,
        vae,
        positive,
        lora_name,
        lora_strength,
        decoder_name,
        task,
        view_resolution,
        view_count,
        seed,
        steps,
        guidance,
        denoise,
        sparse_retention,
        extra_angle_mode="guided",
        extra_denoise=0.55,
        extra_steps=16,
        tiled_decode=True,
        negative=None,
    ):
        import folder_paths
        from .core.conditions import encode_conditions, build_tokens, build_workset
        from .core.models import apply_lora, foreground_decoder, validate_adapter_task
        from .core.sampling import sample_workset
        from .core.decode import decode_views, inspect_views, contact_sheet
        from .extension import decoder_names
        from .core.contracts import ViewCollection

        diag = _active_diag()
        if diag:
            diag.mark("UTX2.inputs", view_resolution=int(view_resolution), view_count=int(view_count),
                      task=task, steps=int(steps), guidance=float(guidance), denoise=float(denoise),
                      sparse_retention=float(sparse_retention), extra_angle_mode=extra_angle_mode,
                      extra_steps=int(extra_steps), extra_denoise=float(extra_denoise),
                      tiled_decode=bool(tiled_decode), lora_strength=float(lora_strength),
                      camera_fit=getattr(source,"camera_fit",None),
                      image_margin_percent=getattr(source,"image_margin_percent",None),
                      image_padding_pixels=getattr(source,"image_padding_pixels",None),
                      reference_zoom=getattr(source,"reference_zoom",None),
                      normal_view_resolution=getattr(source,"normal_resolution",None))
        setup = _diag_stage("UTX2.camera_and_reference_setup", setup_for_generation,
                            source, int(view_resolution), int(view_count))
        policy = setup.runtime

        path = folder_paths.get_full_path_or_raise('loras', lora_name)
        validate_adapter_task(path, task)
        decoder_names()
        decoder_path = folder_paths.get_full_path_or_raise('ultratex', decoder_name)
        decoder, _ = _diag_stage("UTX2.load_foreground_decoder", foreground_decoder, vae, decoder_path)
        patched, _, _ = _diag_stage("UTX2.attach_lora", apply_lora, model, path, task, float(lora_strength))

        first = setup.gbuffers[0]
        conditions, _ = _diag_stage("UTX2.first_pass.vae_encode", encode_conditions, vae, setup.reference, first, policy)
        pack, _ = _diag_stage("UTX2.first_pass.build_tokens", build_tokens, conditions, policy)
        workset = _diag_stage("UTX2.first_pass.build_workset", build_workset, pack, positive, negative, int(seed), 1, task, policy)
        samples, _ = _diag_stage("UTX2.first_pass.sampling", sample_workset, patched, workset, policy, int(steps), float(guidance), float(denoise), "sparse", float(sparse_retention))
        collection, _ = _diag_stage("UTX2.first_pass.vae_decode", decode_views, decoder, samples, policy, bool(tiled_decode))
        del samples, workset, pack, conditions

        if len(setup.gbuffers) == 2:
            second = setup.gbuffers[1]
            guided = extra_angle_mode == "guided" and task == "albedo"
            if guided:
                from .core.bake import prepare_atlas, bake_map, repair_atlas
                from .core.guidance import reproject_atlas
                temp_size = min(2048, max(1024, first.plan.sample_size))
                atlas, _ = _diag_stage("UTX2.second_pass.prepare_atlas", prepare_atlas, setup.context, first, temp_size, policy)
                atlas_rgb, _, seen, conf, holes, _ = _diag_stage("UTX2.second_pass.bake_guide", bake_map,
                    atlas, collection.sets[0], "base_color", policy,
                    gutter=0, blend_mode="detail_preserving"
                )
                guide_atlas, _, _ = _diag_stage("UTX2.second_pass.repair_guide", repair_atlas, atlas_rgb, holes, atlas, "normal_aware_surface", 8)
                initial, protect = _diag_stage("UTX2.second_pass.reproject_guide", reproject_atlas, setup.context, second, guide_atlas, seen, conf)
                del guide_atlas, holes, atlas_rgb, seen, conf, atlas
                second_conditions, _ = _diag_stage("UTX2.second_pass.vae_encode_guided", encode_conditions,
                    vae, setup.reference, second, policy, init_images=initial, keep_masks=protect
                )
                del initial, protect
                actual_denoise = float(extra_denoise)
                actual_steps = int(extra_steps)
            else:
                second_conditions, _ = _diag_stage("UTX2.second_pass.vae_encode_independent", encode_conditions, vae, setup.reference, second, policy)
                actual_denoise = float(denoise)
                actual_steps = int(steps)

            second_pack, _ = _diag_stage("UTX2.second_pass.build_tokens", build_tokens, second_conditions, policy)
            second_workset = _diag_stage("UTX2.second_pass.build_workset", build_workset, second_pack, positive, negative, (int(seed) + 1) % (2**63), 1, task, policy)
            second_samples, _ = _diag_stage("UTX2.second_pass.sampling", sample_workset,
                patched, second_workset, policy, actual_steps, float(guidance), actual_denoise, "sparse", float(sparse_retention)
            )
            rotated, _ = _diag_stage("UTX2.second_pass.vae_decode", decode_views, decoder, second_samples, policy, bool(tiled_decode))
            collection = ViewCollection(collection.sets + rotated.sets)
            del second_conditions, second_pack, second_workset, second_samples, rotated

        batches = []
        labels = []
        for i, views in enumerate(collection.sets):
            active=getattr(views.plan, "active_indices", tuple(range(6)))
            batches.append(inspect_views(collection, i, active, 512)[0])
            labels.extend(f"{'0' if i == 0 else '45'} deg - {views.plan.names[vi]}" for vi in active)
        images = _diag_stage("UTX2.output_prep", torch.cat, batches)
        return io.NodeOutput(collection, images, _diag_stage("UTX2.contact_sheet", contact_sheet, images, labels), setup)


def validate_bake_branch(setup, views, task):
    """Check a branch against ITS OWN generation setup, never another branch's resolution.

    Camera-plan identity includes resolution, so comparing MR plan identities against
    Albedo incorrectly rejects a valid 12-view Albedo + 6-view MR pairing.
    """
    if setup is None or views is None:
        raise ValueError(f"{task} needs its own matching Generate Views setup AND views")
    if len(views.sets) != len(setup.gbuffers):
        raise ValueError(f"{task}: {len(views.sets)*6} views do not match the {len(setup.gbuffers)*6}-view setup")
    if task == "mr" and any(v.task != "mr" for v in views.sets):
        raise ValueError("MR input must come from Generate Views with task=mr")
    if task == "albedo" and any(v.task == "mr" for v in views.sets):
        raise ValueError("Albedo input must not be MR views")
    for i, (vs, gb) in enumerate(zip(views.sets, setup.gbuffers)):
        if vs.plan.identity != gb.plan.identity or vs.gbuffer.plan.identity != gb.plan.identity:
            raise ValueError(f"{task}: camera rig {i} is not from its connected setup. Reconnect that Generate node's setup and views together")
        if vs.plan.geometry_id != setup.context.geometry_id:
            raise ValueError(f"{task}: geometry does not match its generation setup")
    return True


def validate_shared_bake_geometry(albedo_setup, mr_setup):
    """Two branches may differ in views/resolution but must use one geometry/UV atlas."""
    if albedo_setup.context.geometry_id != mr_setup.context.geometry_id:
        raise ValueError("Albedo and MR use different mesh geometry/normals. Connect both Generate nodes to one Prepare source")
    if albedo_setup.context.atlas_id != mr_setup.context.atlas_id:
        raise ValueError("Albedo and MR use different UV layouts. Connect both Generate nodes to one Prepare source")
    return True


SIMPLE_NODES = [UnwrapMeshUV, PrepareTexture, GenerateTextureViews]
