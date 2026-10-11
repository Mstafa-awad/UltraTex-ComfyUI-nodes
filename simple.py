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


def setup_for_generation(source, resolution, view_count):
    from .core.geometry import plan_views
    from .core.render import render_gbuffers
    from .core.conditions import prepare_reference

    resolution = int(resolution)
    view_count = int(view_count)
    if resolution < 256 or resolution > 4096 or resolution % 16:
        raise ValueError("Generate resolution must be a multiple of 16 between 256 and 4096")
    if view_count not in (6, 12):
        raise ValueError("UltraTex supports calibrated 6 or 12 views")
    gbuffers=[]
    for yaw in ((0.0,) if view_count == 6 else (0.0, 45.0)):
        # Render at generation resolution, not at a stale value stored by Prepare.
        plan = plan_views(source.context, max(2048, resolution), resolution, yaw)
        gbuffers.append(render_gbuffers(source.context, plan, source.runtime)[0])
    reference = prepare_reference(source.reference_image, source.reference_mask,
                                  gbuffers[0], source.runtime,
                                  source.mask_polarity, source.alignment)
    return TextureSetup(source.context, tuple(gbuffers), reference, source.runtime)


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
                io.Mask.Input("mask", optional=True), POLICY.Input("runtime", optional=True),
            ], [
                SOURCE.Output(display_name="prepared_source"),
                io.Mesh.Output(display_name="prepared_mesh"),
                io.Image.Output(display_name="normal_views_512"),
                io.Image.Output(display_name="aligned_reference_512"),
                io.Mask.Output(display_name="aligned_mask_512"),
            ], "Prepare expects a UV-ready mesh. Use the standalone UV node first if needed. Resolution and view count belong exclusively to Generate. Preview is fixed at 512px."
        )

    @classmethod
    def execute(cls, mesh, reference, alignment, mask_polarity,
                normal_mode="auto_smooth", mask=None, runtime=None):
        from .core.geometry import plan_views
        from .core.render import render_gbuffers
        from .core.conditions import prepare_reference
        source = prepare_source(mesh, reference, mask, alignment, mask_polarity, runtime, normal_mode)
        # One lightweight 512px diagnostic rig: NOT a generation constraint.
        plan = plan_views(source.context, 512, 512, 0.0)
        gbuffer, _ = render_gbuffers(source.context, plan, source.runtime)
        preview_ref = prepare_reference(source.reference_image, source.reference_mask,
                                        gbuffer, source.runtime, source.mask_polarity,
                                        source.alignment)
        normals = _preview_normals(gbuffer, preview_size=512)
        aligned, aligned_mask = _preview_reference(preview_ref, preview_size=512)
        return io.NodeOutput(source, source.context.mesh, normals, aligned, aligned_mask)


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
                io.Combo.Input("view_count", options=[6, 12], default=6,
                    tooltip="Must match the view count chosen in Prepare."),
                io.Int.Input("seed", default=0, min=0, max=2**63 - 1, control_after_generate=True),
                io.Int.Input("steps", default=4, min=1, max=100),
                io.Float.Input("guidance", default=4.0, min=0.0, max=20.0, step=0.1,
                    tooltip="CFG scale: active only when a different negative CONDITIONING is connected. With no negative input, it does not change sampling."),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("sparse_retention", default=0.20, min=0.01, max=1.0, step=0.01,
                    tooltip="Lower is lighter/faster; higher keeps more tokens in sparse attention."),
                io.Combo.Input("extra_angle_mode", options=["guided", "independent"], default="guided",
                    tooltip="When 12 views are selected, reproject the first pass into the 45-degree pass before refinement."),
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

        setup = setup_for_generation(source, int(view_resolution), int(view_count))
        policy = setup.runtime

        path = folder_paths.get_full_path_or_raise('loras', lora_name)
        validate_adapter_task(path, task)
        decoder_names()
        decoder_path = folder_paths.get_full_path_or_raise('ultratex', decoder_name)
        decoder, _ = foreground_decoder(vae, decoder_path)
        patched, _, _ = apply_lora(model, path, task, float(lora_strength))

        first = setup.gbuffers[0]
        conditions, _ = encode_conditions(vae, setup.reference, first, policy)
        pack, _ = build_tokens(conditions, policy)
        workset = build_workset(pack, positive, negative, int(seed), 1, task, policy)
        samples, _ = sample_workset(patched, workset, policy, int(steps), float(guidance), float(denoise), "sparse", float(sparse_retention))
        collection, _ = decode_views(decoder, samples, policy, bool(tiled_decode))
        del samples, workset, pack, conditions

        if len(setup.gbuffers) == 2:
            second = setup.gbuffers[1]
            guided = extra_angle_mode == "guided" and task == "albedo"
            if guided:
                from .core.bake import prepare_atlas, bake_map, repair_atlas
                from .core.guidance import reproject_atlas
                temp_size = min(2048, max(1024, first.plan.sample_size))
                atlas, _ = prepare_atlas(setup.context, first, temp_size, policy)
                atlas_rgb, _, seen, conf, holes, _ = bake_map(
                    atlas, collection.sets[0], "base_color", policy,
                    gutter=0, blend_mode="detail_preserving"
                )
                guide_atlas, _, _ = repair_atlas(atlas_rgb, holes, atlas, "normal_aware_surface", 8)
                initial, protect = reproject_atlas(setup.context, second, guide_atlas, seen, conf)
                del guide_atlas, holes, atlas_rgb, seen, conf, atlas
                second_conditions, _ = encode_conditions(
                    vae, setup.reference, second, policy, init_images=initial, keep_masks=protect
                )
                del initial, protect
                actual_denoise = float(extra_denoise)
                actual_steps = int(extra_steps)
            else:
                second_conditions, _ = encode_conditions(vae, setup.reference, second, policy)
                actual_denoise = float(denoise)
                actual_steps = int(steps)

            second_pack, _ = build_tokens(second_conditions, policy)
            second_workset = build_workset(second_pack, positive, negative, (int(seed) + 1) % (2**63), 1, task, policy)
            second_samples, _ = sample_workset(
                patched, second_workset, policy, actual_steps, float(guidance), actual_denoise, "sparse", float(sparse_retention)
            )
            rotated, _ = decode_views(decoder, second_samples, policy, bool(tiled_decode))
            collection = ViewCollection(collection.sets + rotated.sets)
            del second_conditions, second_pack, second_workset, second_samples, rotated

        batches = []
        labels = []
        for i, views in enumerate(collection.sets):
            batches.append(inspect_views(collection, i, tuple(range(6)), 512)[0])
            labels.extend(f"{'0' if i == 0 else '45'} deg - {name}" for name in views.plan.names)
        images = torch.cat(batches)
        return io.NodeOutput(collection, images, contact_sheet(images, labels), setup)


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


def bake_texture(albedo_setup, albedo_views, texture_size=2048, fill_unseen=True,
                 mr_setup=None, mr_views=None, blend_mode="detail_preserving", repair_mode="normal_aware_surface"):
    """Bake Albedo and MR independently onto ONE native mesh and UV atlas.

    The two setups carry different camera rigs / image sizes. Reuse expensive atlas
    rasterization, but swap the branch-specific G-buffer used by bake_map. This
    preserves each view's own depth maps and camera matrices at its own resolution.
    """
    from .core.bake import prepare_atlas, bake_map, repair_atlas
    from .core.materials import apply_materials, create_file

    validate_bake_branch(albedo_setup, albedo_views, "albedo")
    if (mr_setup is None) != (mr_views is None):
        raise ValueError("Connect BOTH MR setup and MR views (or neither for Albedo-only baking)")
    if mr_setup is not None:
        validate_bake_branch(mr_setup, mr_views, "mr")
        validate_shared_bake_geometry(albedo_setup, mr_setup)

    atlas, _ = prepare_atlas(albedo_setup.context, albedo_setup.gbuffers[0],
                             int(texture_size), albedo_setup.runtime)
    tasks = [("base_color", albedo_setup, albedo_views)]
    if mr_views is not None:
        tasks.extend((("roughness", mr_setup, mr_views), ("metallic", mr_setup, mr_views)))
    maps = {}
    infos = {}
    unseen = None
    confidence_out = None
    for semantic, branch_setup, collection in tasks:
        # Atlas UV raster is shared; only the view-specific camera/depth context
        # changes. Crucial when MR is 1024/6 while Albedo is 2048/12.
        branch_atlas = replace(atlas, gbuffer=branch_setup.gbuffers[0])
        image, info, observed, confidence, holes, _ = bake_map(
            branch_atlas, collection.sets[0], semantic, branch_setup.runtime,
            gutter=0, blend_mode=blend_mode,
            extra_views=tuple((vs, tuple(range(6))) for vs in collection.sets[1:]),
        )
        image, _, _ = repair_atlas(image, holes, atlas,
                                   repair_mode if fill_unseen else "none", 8)
        maps[semantic] = torch.from_numpy(image[None])
        infos[semantic] = info
        if semantic == "base_color":
            unseen = torch.from_numpy(holes[None])
            confidence_out = torch.from_numpy(confidence[None])

    # Provide preview images even without an MR branch. Do not attach fake MR
    # textures to the mesh unless a material branch actually ran.
    roughness_out = maps.get("roughness", torch.full_like(maps["base_color"], 0.8))
    metallic_out = maps.get("metallic", torch.zeros_like(maps["base_color"]))
    actual_maps = {name: maps[name] for name in ("base_color", "roughness", "metallic") if name in maps}
    mesh, _ = apply_materials(albedo_setup.context.mesh, map_infos=infos, **actual_maps)
    return mesh, maps["base_color"], roughness_out, metallic_out, create_file(mesh), unseen, confidence_out


class BakeTexture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "UTXBakeTextureV26", "3. Smart UV Bake — Separate Albedo / MR Setups",
            [
                SETUP.Input("albedo_setup"),
                VIEWS.Input("albedo_views"),
                io.Int.Input("texture_size", default=2048, min=512, max=8192, step=16),
                io.Boolean.Input("fill_unseen", default=True,
                    tooltip="Estimate colors for hidden surfaces (does not generate new detail)."),
                io.Combo.Input("blend_mode", options=["detail_preserving", "reference", "smooth", "legacy"], default="detail_preserving"),
                io.Combo.Input("repair_mode", options=["normal_aware_surface", "nearest_surface", "nearest_in_chart", "none"], default="normal_aware_surface"),
                SETUP.Input("mr_setup", optional=True),
                VIEWS.Input("mr_views", optional=True),
            ],
            [
                io.Mesh.Output(),
                io.Image.Output(display_name="base_color"),
                io.Image.Output(display_name="roughness"),
                io.Image.Output(display_name="metallic"),
                io.File3DGLB.Output(),
                io.Mask.Output(display_name="unseen_before_fill"),
                io.Mask.Output(display_name="projection_confidence"),
            ],
            "Connect each Generate Views node's setup AND views to its corresponding branch. Albedo 12/2048 plus MR 6/1024 supported. Uses one shared mesh/UV layout.",
        )

    @classmethod
    def execute(cls, albedo_setup, albedo_views, texture_size, fill_unseen,
                blend_mode="detail_preserving", repair_mode="normal_aware_surface",
                mr_setup=None, mr_views=None):
        return io.NodeOutput(*bake_texture(
            albedo_setup, albedo_views, int(texture_size), fill_unseen,
            mr_setup, mr_views, blend_mode, repair_mode,
        ))


SIMPLE_NODES = [UnwrapMeshUV, PrepareTexture, GenerateTextureViews, BakeTexture]
