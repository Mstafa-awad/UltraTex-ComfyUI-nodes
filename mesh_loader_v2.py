"""UTX v2 optional local mesh importer. Does not alter the original UltraTex pipeline.

This node intentionally emits a native ComfyUI MESH, not trimesh, and does not
rewrite UVs or geometry unless source UV coordinates use OBJ's V-up convention.
FBX uses a separate locally installed Blender executable to convert to GLB.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from comfy_api.latest import Types, io

_MESH_EXTS = {".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl", ".off", ".3mf", ".dae"}
_CUSTOM = "(enter a custom path)"


def _input_dir() -> Path:
    import folder_paths
    return Path(folder_paths.get_input_directory()).resolve()


def _available_files() -> list[str]:
    root = _input_dir()
    found = []
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in _MESH_EXTS:
                found.append(path.relative_to(root).as_posix())
                if len(found) >= 5000:
                    break
    return [_CUSTOM] + sorted(found, key=str.casefold)


def _resolve_file(choice: str, file_path: str) -> Path:
    entered = file_path.strip().strip('"')
    root = _input_dir()
    if entered:
        selected = Path(entered).expanduser()
        if not selected.is_absolute():
            selected = root / selected
    elif choice != _CUSTOM:
        selected = root / choice
        # Dropdown choices must remain inside input/; explicit absolute custom paths are allowed.
        try:
            selected.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("Selected filename escapes the ComfyUI input folder") from exc
    else:
        raise ValueError("Select a mesh in ComfyUI/input or enter an absolute/relative file_path")
    selected = selected.resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"Mesh file not found: {selected}")
    if selected.suffix.lower() not in _MESH_EXTS:
        raise ValueError(f"Unsupported file extension: {selected.suffix}. Supported: {', '.join(sorted(_MESH_EXTS))}")
    return selected


_BLENDER_FBX_CONVERT = r'''
import bpy,sys
argv=sys.argv[sys.argv.index('--')+1:]
source,dest=argv[0],argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.import_scene.fbx(filepath=source)
except (AttributeError,RuntimeError):
    bpy.ops.wm.fbx_import(filepath=source)
bpy.ops.export_scene.gltf(filepath=dest,export_format='GLB',export_apply=True)
'''


def _load_scene(path: Path, blender_path: str = ""):
    import trimesh
    if path.suffix.lower() not in {".fbx", ".dae"}:
        try:
            return trimesh.load_scene(str(path), process=False)
        except Exception as exc:
            raise ValueError(f"Unable to import {path.name}: {exc}") from exc
    # Optional Blender is the reliable FBX/DAE bridge; many trimesh installations
    # cannot parse FBX. This conversion is temporary and never edits the source.
    blender = blender_path.strip().strip('"') or shutil.which("blender")
    if not blender:
        raise RuntimeError(f"{path.suffix.upper()} import requires Blender. Set blender_path to blender.exe, or convert to GLB in Blender first.")
    with tempfile.TemporaryDirectory(prefix="utx_mesh_import_") as tmp:
        tmp = Path(tmp)
        script = tmp / "convert.py"
        output = tmp / "converted.glb"
        script.write_text(_BLENDER_FBX_CONVERT, encoding="utf-8")
        cmd = [str(blender), "--background", "--factory-startup", "--python", str(script), "--", str(path), str(output)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Blender conversion failed: {exc}") from exc
        if result.returncode != 0 or not output.is_file():
            details = (result.stderr or result.stdout)[-1200:]
            raise RuntimeError(f"Blender could not convert {path.name}. {details}")
        return trimesh.load_scene(str(output), process=False)


def _mesh_data(scene, preserve_uv: bool, suffix: str):
    """Return mesh arrays, preserving all geometry instances and per-vertex seams."""
    import trimesh
    if isinstance(scene, trimesh.Trimesh):
        parts = [(scene,np.eye(4))]
    elif isinstance(scene, trimesh.Scene):
        # Scene.dump() copies meshes without their authored normal cache in
        # supported Trimesh versions. Preserve the source fields and transform
        # each instance explicitly; normals need inverse-transpose for scale.
        parts = [(scene.geometry[name],matrix) for node in scene.graph.nodes_geometry
                 for matrix,name in [scene.graph[node]]]
    else:
        raise ValueError("The file does not contain a supported triangular surface mesh")
    vertices, faces, uvs, normals = [], [], [], []
    uv_valid = bool(preserve_uv)
    offset = 0
    component_count = 0
    for part,transform in parts:
        if not isinstance(part, trimesh.Trimesh) or len(part.vertices) == 0 or len(part.faces) == 0:
            continue
        if not (np.asarray(part.faces).ndim == 2 and np.asarray(part.faces).shape[1] == 3):
            raise ValueError("Only triangular face meshes are supported")
        v = np.asarray(part.vertices, dtype=np.float32)
        f = np.asarray(part.faces, dtype=np.int64)
        transform=np.asarray(transform,np.float64)
        if transform.shape!=(4,4) or not np.isfinite(transform).all():
            raise ValueError('Mesh instance transform is invalid')
        try:normal_matrix=np.linalg.inv(transform[:3,:3])
        except np.linalg.LinAlgError as exc:raise ValueError('Mesh instance transform is singular') from exc
        v=np.asarray(trimesh.transform_points(v,transform),np.float32)
        if np.linalg.det(transform[:3,:3])<0:f=f[:,::-1]
        if not (np.isfinite(v).all() and f.min() >= 0 and f.max() < len(v)):
            raise ValueError("Mesh has invalid/nonfinite vertices or face indices")
        vertices.append(v)
        faces.append(f + offset)
        if preserve_uv:
            raw_uv = getattr(getattr(part, "visual", None), "uv", None)
            if raw_uv is None:
                uv_valid = False
            else:
                uv = np.asarray(raw_uv, dtype=np.float32)
                if uv.shape != (len(v), 2) or not np.isfinite(uv).all():
                    uv_valid = False
                else:
                    if suffix in {".obj", ".glb", ".gltf"}:
                        # Trimesh exposes image UVs as V-up, including GLB/glTF
                        # imports. Native ComfyUI MESH and its exporter use V-down.
                        uv = uv.copy()
                        uv[:, 1] = 1.0 - uv[:, 1]
                    uvs.append(uv)
        raw_normals = np.asarray(part.vertex_normals, dtype=np.float64)@normal_matrix
        raw_normals/=np.maximum(np.linalg.norm(raw_normals,axis=1,keepdims=True),1e-20)
        raw_normals=np.asarray(raw_normals,np.float32)
        if raw_normals.shape != (len(v), 3) or not np.isfinite(raw_normals).all():
            raise ValueError("Mesh normals are invalid")
        normals.append(raw_normals)
        offset += len(v)
        component_count += 1
    if not component_count:
        raise ValueError("No surface triangles found in file")
    if preserve_uv and (not uv_valid or len(uvs) != component_count):
        raise ValueError("Copy UVs was selected, but a mesh part has no valid per-vertex UVs. Uncheck Copy UVs, then use UTX UV Unwrap Mesh.")
    merged_vertices = np.ascontiguousarray(np.concatenate(vertices), dtype=np.float32)
    merged_faces = np.ascontiguousarray(np.concatenate(faces).astype(np.int32), dtype=np.int32)
    merged_normals = np.ascontiguousarray(np.concatenate(normals), dtype=np.float32)
    merged_uvs = np.ascontiguousarray(np.concatenate(uvs), dtype=np.float32) if preserve_uv else None
    return merged_vertices, merged_faces, merged_uvs, merged_normals, component_count


class LoadMeshV2(io.ComfyNode):
    """Load OBJ, GLB, glTF, PLY, STL, OFF, 3MF; FBX/DAE via optional Blender."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UTXLoadMeshV2",
            display_name="UTX 0. Load Mesh (v2)",
            category="UltraTex Modular",
            inputs=[
                io.Combo.Input("mesh_file", options=_available_files(), default=_CUSTOM),
                io.String.Input("file_path", default="", tooltip="Absolute path, or path relative to ComfyUI/input. Overrides the dropdown."),
                io.Boolean.Input("copy_uvs", default=True, tooltip="Preserve imported UV coordinates. Disable to unwrap later."),
                io.String.Input("blender_path", default="", optional=True, tooltip="Optional Blender executable for FBX/DAE import."),
            ],
            outputs=[io.Mesh.Output(display_name="mesh"), io.String.Output(display_name="details")],
            description="Import a local 3D file into native ComfyUI MESH. UV seam vertices are kept when Copy UVs is enabled. FBX/DAE need Blender.",
        )

    @classmethod
    def execute(cls, mesh_file, file_path, copy_uvs, blender_path=""):
        import torch
        path = _resolve_file(mesh_file, file_path)
        scene = _load_scene(path, blender_path)
        v, f, uv, n, parts = _mesh_data(scene, bool(copy_uvs), ".glb" if path.suffix.lower() in {".fbx", ".dae"} else path.suffix.lower())
        result = Types.MESH(
            vertices=torch.from_numpy(v.copy()).unsqueeze(0),
            faces=torch.from_numpy(f.copy()).unsqueeze(0),
            uvs=torch.from_numpy(uv.copy()).unsqueeze(0) if uv is not None else None,
            normals=torch.from_numpy(n.copy()).unsqueeze(0),
        )
        return io.NodeOutput(result, f"Loaded {path.name}: {len(v):,} vertices / {len(f):,} faces / {parts} part(s); UV copied: {uv is not None}. Mesh materials/textures are not imported; use UltraTex material and baking nodes.")
