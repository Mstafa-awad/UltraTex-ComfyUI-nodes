from __future__ import annotations

import copy
import json
import struct

import numpy as np

from .contracts import MeshContext, ViewPlan
from .runtime import fingerprint, interrupt, measure, ram_preflight

GRID_ORDER = ((0, 1, 3), (2, 4, 5))
FOCAL = 6400. / 4096.
# Canonical TexVerse cameras, from the MIT upstream wrapper (see THIRD_PARTY_NOTICES).
POSES = np.array([
    [[1, 0, 0, 0], [0, 0, -1, -1.67], [0, 1, 0, 0]],
    [[0, 0, -1, -1.67], [-1, 0, 0, 0], [0, 1, 0, 0]],
    [[-1, 0, 0, 0], [0, 0, 1, 1.67], [0, 1, 0, 0]],
    [[0, 0, 1, 1.67], [1, 0, 0, 0], [0, 1, 0, 0]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.67]],
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -1.67]],
], dtype=np.float32)
GLTF_TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def numpy_cpu(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def mesh_arrays(mesh):
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError("UltraTex requires native ComfyUI MESH. Convert FILE_3D with UTX File to Mesh.")
    if mesh.vertices.ndim != 3 or mesh.vertices.shape[0] != 1 or mesh.faces.shape[0] != 1:
        raise ValueError("UltraTex supports exactly one mesh per work item. Split the mesh batch first.")
    vc, fc = getattr(mesh, "vertex_counts", None), getattr(mesh, "face_counts", None)
    if (vc is None) != (fc is None):
        raise ValueError("vertex_counts and face_counts must be provided together")
    n = int(vc[0]) if vc is not None else mesh.vertices.shape[1]
    f = int(fc[0]) if fc is not None else mesh.faces.shape[1]
    if not (0 < n <= mesh.vertices.shape[1] and 0 < f <= mesh.faces.shape[1]):
        raise ValueError("Empty mesh or invalid padded mesh counts")
    v = numpy_cpu(mesh.vertices[0, :n], np.float32)
    faces = numpy_cpu(mesh.faces[0, :f])
    if v.shape[1] != 3 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Expected 3D vertices and integer triangle indices")
    if not np.isfinite(v).all() or faces.min() < 0 or faces.max() >= n:
        raise ValueError("Nonfinite vertices or out-of-range face indices")
    if n >= 2**31:
        raise ValueError("Mesh exceeds the renderer's 32-bit index limit")
    faces = np.asarray(faces, dtype=np.int32)
    uv = None if getattr(mesh, "uvs", None) is None else numpy_cpu(mesh.uvs[0, :n], np.float32)
    normals = None if getattr(mesh, "normals", None) is None else numpy_cpu(mesh.normals[0, :n], np.float32)
    if uv is not None and (uv.shape != (n, 2) or not np.isfinite(uv).all()):
        raise ValueError("UVs must be finite per-vertex pairs matching the real vertex count")
    return v, faces, uv, normals


def compute_normals(vertices, faces, normals=None):
    """Preserve authored normals; repair missing normals across exact UV duplicates.

    UV unwrapping splits vertices and some native MESH nodes drop normals. Those
    splits must not become lighting/baking creases. Group positions for the normal
    calculation only: vertices, faces, UVs and intentional valid hard normals stay
    untouched. A single invalid normal must never discard all authored normals.
    """
    result = np.zeros_like(vertices, dtype=np.float32)
    valid = np.zeros(len(vertices), dtype=bool)
    if normals is not None and normals.shape == vertices.shape:
        lengths = np.linalg.norm(normals, axis=1)
        valid = np.isfinite(normals).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-10)
        result[valid] = normals[valid] / lengths[valid, None]
        if valid.all():
            return result
    _, groups = np.unique(vertices, axis=0, return_inverse=True)
    acc = np.zeros((int(groups.max()) + 1, 3), dtype=np.float32)
    for start in range(0, len(faces), 131072):
        interrupt()
        fi = faces[start:start + 131072]
        p = vertices[fi]
        fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        for c in range(3):
            np.add.at(acc, groups[fi[:, c]], fn)
    missing = ~valid
    result[missing] = acc[groups[missing]]
    lengths = np.linalg.norm(result, axis=1)
    still = lengths < 1e-10
    if still.any():
        if (~still).any():
            from scipy.spatial import cKDTree
            _, nearest = cKDTree(vertices[~still]).query(vertices[still])
            result[still] = result[~still][nearest]
        else:
            result[still] = (0., 0., 1.)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


def condition_normals(vertices, faces, authored=None, mode="auto_smooth"):
    """Normal conditioning never changes triangles, UVs, or vertex positions.

    The welded normal is accumulated using triangle *areas* at coincident
    positions. This removes invisible UV-split creases in the normal G-buffer.
    For a clearly intentional sharp normal (>27 degree deviation), auto_smooth
    preserves the authored direction; use smooth to override everything.
    """
    if mode not in ("auto_smooth", "preserve", "smooth"):
        raise ValueError("normal_mode must be auto_smooth, preserve, or smooth")
    base = compute_normals(vertices, faces, authored)
    if mode == "preserve":
        return base
    welded = compute_normals(vertices, faces, None)
    if mode == "smooth" or authored is None:
        return welded
    # At ordinary triangulation/UV splits the source normal can be slightly
    # different from the continuous surface. Keep deliberately sharp normals.
    similarity = np.einsum("ij,ij->i", base, welded)
    sharp = similarity < np.cos(np.deg2rad(27.))
    welded[sharp] = base[sharp]
    return welded


def prepare_geometry(mesh, uv_convention="v_down", policy=None, normal_mode="auto_smooth"):
    """Prepare a non-destructive normalized mesh.

    auto_smooth detects normals that disagree with area-weighted welded normals
    only by a small angle (common per-triangle/UV-split artifacts). It preserves
    deliberately hard normals at edges deviating by > 27 degrees. A forced
    smooth mode is also available for diagnostic face/cloth comparisons.
    """

    with measure("prepare_geometry") as report:
        v, faces, uv, n = mesh_arrays(mesh)
        if policy:
            ram_preflight(v.nbytes * 3 + faces.nbytes, policy.system_headroom_mb)
        used = np.zeros(len(v), dtype=bool)
        for start in range(0, len(faces), 131072):
            used[faces[start:start + 131072].ravel()] = True
        lower, upper = v[used].min(0), v[used].max(0)
        centre = (lower + upper) * .5
        normalized = v - centre
        radius = float(np.sqrt(np.einsum("ij,ij->i", normalized[used], normalized[used]).max()))
        if radius < 1e-12:
            raise ValueError("Mesh has zero spatial extent")
        scale = .5 / radius
        normalized *= scale
        n = condition_normals(v, faces, n, normal_mode)
        gid = fingerprint("geometry-v2-normal-policy", v, faces, n, normal_mode)
        aid = fingerprint("atlas-v1", v, faces, uv, uv_convention) if uv is not None else ""
        for a in (normalized, n):
            a.flags.writeable = False
        context = MeshContext(mesh, normalized, faces, n, uv, gid, aid,
                              tuple(float(x) for x in centre), scale, uv_convention)
        report.update(vertices=len(v), faces=len(faces), has_uv=uv is not None,
                      bounds=[lower.tolist(), upper.tolist()], geometry_id=gid, atlas_id=aid,
                      material_provenance="unknown for native MESH; source primitive metadata is unavailable")
    return context, report


def plan_views(context, render_size=2048, sample_size=1024, yaw=0., elevation=0.):
    if sample_size % 16 or sample_size < 64 or sample_size > 4096:
        raise ValueError("Sample size must be a multiple of 16, between 64 and 4096")
    if render_size < 64 or render_size > 4096:
        raise ValueError("Render size must be between 64 and 4096")
    a, e = np.radians(yaw), np.radians(elevation)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]], np.float32)
    rot = rx @ ry
    frame = np.eye(4, dtype=np.float32)
    frame[:3, :3] = GLTF_TO_BLENDER @ rot
    matrices = []
    for pose in POSES:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3] = pose
        matrices.append(tuple(tuple(float(v) for v in row) for row in np.linalg.inv(c2w) @ frame))
    identity = fingerprint("views-v1", context.geometry_id, matrices, render_size, sample_size)
    return ViewPlan(context.geometry_id, render_size, sample_size, tuple(matrices), identity,
                    tuple(tuple(float(v) for v in row) for row in rot))


def validate_uv(context):
    """Accept a tiny number of collapsed triangles, not broken UV layouts.

    Atlas rasterization cannot give zero-area UV triangles distinct texels. We
    warn and leave those texels to downstream hole handling rather than abort
    millions of otherwise valid faces. No mesh vertices/UV coordinates change.
    """
    uv = context.uvs
    if uv is None:
        raise ValueError("This mesh has no UVs. Run the stock Unwrap Mesh UVs node explicitly before baking.")
    if np.any(uv < -1e-6) or np.any(uv > 1 + 1e-6):
        raise ValueError("UVs outside [0,1] are unsupported (UDIM/repeat layouts need an explicit adapter).")
    bad = 0
    for start in range(0, len(context.faces), 131072):
        tri = uv[context.faces[start:start + 131072]]
        a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        bad += int(np.count_nonzero(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) < 1e-14))
    # Allow <=0.02% on large meshes (a 16-triangle floor), but never >1% on
    # tiny meshes. A wholly invalid two-triangle asset must still be rejected.
    count = len(context.faces)
    allowed = min(max(16, int(count * 0.0002)), int(count * 0.01))
    if bad > allowed:
        raise ValueError(f"{bad} triangles have degenerate UVs (tolerance {allowed}). "
                         "Repair or explicitly unwrap the mesh; excessive loss is unsafe for baking.")
    return bad


def glb_json(data):
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("Expected a binary glTF 2.0 file")
    _, version, length = struct.unpack_from("<4sII", data)
    size, kind = struct.unpack_from("<II", data, 12)
    if version != 2 or length != len(data) or kind != 0x4E4F534A or 20 + size > length:
        raise ValueError("Invalid GLB header or JSON chunk")
    return json.loads(data[20:20 + size]), 20 + size


def file_to_mesh(file3d):
    from comfy_extras.nodes_mesh_io import Get3DComponents
    if file3d is None:
        raise ValueError("Select a GLB in Load 3D before converting it to MESH")
    data=file3d.get_bytes()
    document, binary_offset = glb_json(data)
    primitives = [p for m in document.get("meshes", []) for p in m.get("primitives", [])]
    materials = {p.get("material", -1) for p in primitives}
    if len(materials) > 1:
        raise ValueError("Multi-material GLB: atlas/merge materials explicitly before import; native import loses assignments.")
    if document.get("skins") or document.get("animations") or any("targets" in p for p in primitives):
        raise ValueError("Skinning, animation, and morph targets are unsupported by this static-mesh adapter.")
    if any("TEXCOORD_1" in p.get("attributes", {}) for p in primitives):
        raise ValueError("Multiple UV sets require an explicit adapter")
    index=next(iter(materials),-1)
    material=document.get("materials",[])[index] if index>=0 else {}
    def validate_texture_records(value):
        if isinstance(value,dict):
            if "KHR_texture_transform" in value.get("extensions",{}):
                raise ValueError("Bake texture transforms into UVs before GLB import")
            if value.get("texCoord",0)!=0:
                raise ValueError("Only TEXCOORD_0 textures can be imported")
            for child in value.values():
                validate_texture_records(child)
        elif isinstance(value,list):
            for child in value:
                validate_texture_records(child)
    validate_texture_records(material)
    pbr=material.get("pbrMetallicRoughness",{})
    ao=material.get("occlusionTexture")
    if ao and ao.get("index")!=pbr.get("metallicRoughnessTexture",{}).get("index"):
        raise ValueError("Pack standalone AO into ORM before import, or connect its IMAGE to Apply Materials")
    opacity=None
    mode=material.get("alphaMode","OPAQUE")
    base_texture=pbr.get("baseColorTexture")
    if mode!="OPAQUE" and base_texture is not None:
        from io import BytesIO
        from PIL import Image
        import torch
        texture=document["textures"][base_texture["index"]]
        image=document["images"][texture["source"]]
        if "bufferView" not in image:
            raise ValueError("Embed alpha textures in the GLB before import")
        view=document["bufferViews"][image["bufferView"]]
        if view.get("buffer",0)!=0 or binary_offset+8>len(data):
            raise ValueError("Unsupported alpha texture buffer")
        length,kind=struct.unpack_from("<II",data,binary_offset)
        if kind!=0x004E4942:
            raise ValueError("Expected an embedded GLB binary chunk for alpha")
        start=binary_offset+8+view.get("byteOffset",0)
        end=start+view["byteLength"]
        if end>min(len(data),binary_offset+8+length):
            raise ValueError("Alpha texture buffer is out of range")
        with Image.open(BytesIO(data[start:end])) as image:
            opacity=torch.from_numpy(np.array(image.convert("RGBA").getchannel("A"),np.float32)/255.)[None]
    result = Get3DComponents.execute(file3d)
    mesh=result[0]
    mesh.material=dict(getattr(mesh,"material",None) or {})
    mesh.material.update(utx_alpha_mode=mode,utx_alpha_cutoff=material.get("alphaCutoff",.5))
    if opacity is not None:
        mesh.material["utx_opacity"]=opacity
    return mesh, {"stage": "file_to_mesh", "primitives": len(primitives), "materials": len(materials)}
