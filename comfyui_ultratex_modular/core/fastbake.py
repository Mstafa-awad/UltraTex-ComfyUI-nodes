"""UTX Fast Surface Texture Baker: production projection, fusion and completion.

Independent implementation (NumPy/SciPy/PyTorch only). No OpenGL context, no
triangle-ID matching, no per-texel Python loops and no disk spill of large
intermediates. Pipeline:

1. ``rasterize_uv``: vectorised software UV rasterisation. Every occupied texel
   stores its triangle and barycentric coordinates. Triangles too small to own
   a texel centre still receive the texel under their UV centroid
   (conservative coverage), so tiny islands never inherit a neighbour's colour.
   Cached per (atlas identity, size) and shared by Base Color and MR.
2. ``_project``: each texel's 3D point is projected into every generated view.
   Visibility is a *plane-consistent depth test* against the exact depth
   buffer that conditioned the generation: for the four depth pixels around
   the projection, the depth the texel's own triangle plane would have at that
   pixel's ray is predicted and compared. Same surface = within ~1.5 depth
   pixels of the plane, independent of triangle identity, so small triangles
   and triangle edges are not rejected, while a real occluder in front is.
   View pixels that straddle a silhouette or depth jump are excluded from colour
   sampling (white rims / colour mixtures); sampling is mask-normalised and
   mip-aware.
3. Fusion: quality q = cos(normal, view) * visibility * distance-to-image-edge.
   Albedo views first get a robust per-view colour gain (clamped to +-15 %).
   Two bands: detail from weights q**8 (6 views) / q**14 (12 views), colour from
   q**2, combined on the mesh surface (smooth hand-over, no hard camera switch).
4. ``complete_texels``: texels without a trustworthy observation are completed
   on the *surface*, not in the image: a harmonic (membrane) fill on the welded
   mesh graph (continuous across UV seams, never jumping to unrelated surfaces),
   refined in a band of texels by chart-restricted SOR relaxation so the fill
   meets the observed texels without a visible border. Weak grazing-angle
   observations are blended with the completion instead of being trusted blindly.
5. ``to_atlas``: occupied texels are written, gutters are filled from the
   nearest occupied texel (never overwrites an occupied texel).

Detail mode (``detail=True``, Base Color only, used by ``upscale_mode=fast_detail``)
changes ONLY how the colour of an observed texel is estimated; visibility,
confidence, coverage, completion and gutters are the same code paths:

* Reconstruction: where a texel magnifies the view (atlas finer than the view,
  the normal case for a 4096 atlas from 2048 views) and the whole 4x4 window is
  valid, the view is sampled with a cubic kernel (Keys a=-0.75) instead of the
  bilinear tent, with an anti-ringing clamp to the 2x2 nearest view pixels.
  Bilinear magnification keeps 57 % of a 0.4 cycle/px detail, the cubic kernel
  75 %; the clamp removes overshoot halos at edges.
* Fine-band selection: every view sample is split in VIEW space into its fine band
  H = cubic sample - G(1 px) (mask-normalised, so silhouettes/depth edges never
  bleed) and the rest. The rest goes through the unchanged two-band fusion; H is
  fused with weights (q/0.3)**48, i.e. the finest detail comes from essentially
  one camera with a narrow hand-over, so misregistered generated views do not
  average their fine details into blur and double edges. The result is bounded
  by the fused colour and the best camera's own sample (no new dark/bright rims).

Projection, visibility, fusion and relaxation run in PyTorch (CUDA when
available, otherwise CPU), chunked; the raster, splats and sparse solves use
NumPy/SciPy.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .geometry import FOCAL

NEAR = 0.05
# Visibility: same-surface tolerance = TOL_ABS + TOL_PX * (world size of one depth pixel)
TOL_ABS = 1e-4
TOL_PX = 1.5
COS_MIN = 0.02
# Fusion weight exponents (q in [0,1]): detail band ~ best view, colour band blends widely.
FUSION_POWER = 8.0          # sharp band, up to 6 views (cameras 90 degrees apart)
FUSION_POWER_MANY = 14.0    # sharp band, 12 views (cameras 45 degrees apart)
FUSION_POWER_BROAD = 2.0    # low-frequency colour band
# Observation confidence ramp on the best view quality.
Q_LO = 0.06
Q_HI = 0.25
EDGE_FLOOR = 0.05
GAIN_LIMIT = 0.15
BLEND_BAND = 16             # texels over which a completed region is relaxed onto observed texels
# Detail mode (fast_detail): fine band split in view pixels and its fusion power.
DETAIL_SIGMA_FINE = 1.0     # fine band H = cubic sample - G(1 view px); the rest uses the standard fusion
DETAIL_POWER_FINE = 48.0    # fine band: near single-camera selection with a narrow hand-over
DETAIL_Q_REF = 0.3          # q / DETAIL_Q_REF before the power keeps float32 weights normal for q >= Q_LO
DETAIL_LOD_MAX = 0.05       # cubic reconstruction only where the texel magnifies the view (lod ~ 0)
DETAIL_SKIP_RATIO = 0.5     # skip the cubic where q < 0.5 * best q so far: its fine weight is < 0.5**48
DETAIL_RANGE_SLACK = 0.004  # linear: fused + fine stays within [min, max](fused, best camera sample) +- slack


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

class StageTimer:
    def __init__(self):
        self.stages = OrderedDict()
        self._t = time.perf_counter()
        self._start = self._t

    def lap(self, name):
        now = time.perf_counter()
        self.stages[name] = round(self.stages.get(name, 0.0) + now - self._t, 4)
        self._t = now

    def total(self):
        return round(time.perf_counter() - self._start, 4)


def _interrupt():
    try:
        import comfy.model_management as mm
    except Exception:
        return
    mm.throw_exception_if_processing_interrupted()


def _void_rows(a):
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def _weld(keys):
    """Unique-row inverse index for an integer key matrix."""
    _, inverse = np.unique(_void_rows(keys), return_inverse=True)
    inverse = np.asarray(inverse, np.int64).ravel()
    return inverse, (int(inverse.max()) + 1) if inverse.size else 0


def srgb_to_linear_np(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb_np(x):
    x = np.clip(np.asarray(x, np.float32), 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


# --------------------------------------------------------------------------------------
# topology (cached with the raster)
# --------------------------------------------------------------------------------------

@dataclass
class SurfaceTopology:
    faces: np.ndarray          # (F,3) int64 vertex ids of the baked mesh
    weld: np.ndarray           # (V,) id of the position-welded vertex
    n_weld: int
    weld_faces: np.ndarray     # (F,3) int64
    weld_pos: np.ndarray       # (Vw,3) float64
    face_chart: np.ndarray     # (F,) int32, UV island id (>=1)
    n_charts: int
    face_normal: np.ndarray    # (F,3) float32 unit geometric normal (0 if degenerate)
    face_area: np.ndarray      # (F,) float64 3D area
    adjacency: object          # scipy csr (Vw,Vw) symmetric positive edge weights


def build_topology(vertices, faces, uvs):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    qpos = np.round(v * float(2 ** 20)).astype(np.int64)
    weld, nw = _weld(qpos)
    counts = np.bincount(weld, minlength=nw).astype(np.float64)
    weld_pos = np.stack([np.bincount(weld, weights=v[:, k], minlength=nw) for k in range(3)], 1)
    weld_pos /= np.maximum(counts, 1)[:, None]
    weld_faces = weld[f]

    # UV islands: faces connected through vertices that share BOTH position and UV.
    # Robust for native split-seam meshes, triangle soups and split-normal meshes.
    if uvs is not None:
        quv = np.round(np.asarray(uvs, np.float64) * float(2 ** 22)).astype(np.int64)
        cid, nc = _weld(np.concatenate([qpos, quv], 1))
    else:
        cid, nc = weld, nw
    cf = cid[f]
    rows = np.concatenate([cf[:, 0], cf[:, 1], cf[:, 2]])
    cols = np.concatenate([cf[:, 1], cf[:, 2], cf[:, 0]])
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(nc, nc)).tocsr()
    _, labels = connected_components(graph, directed=False)
    face_chart = labels[cf[:, 0]].astype(np.int32) + 1
    # compact chart ids
    uniq, face_chart = np.unique(face_chart, return_inverse=True)
    face_chart = face_chart.astype(np.int32) + 1

    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    norm = np.linalg.norm(cross, axis=1)
    face_area = 0.5 * norm
    face_normal = np.where(norm[:, None] > 1e-20, cross / np.maximum(norm, 1e-30)[:, None], 0.0).astype(np.float32)

    # Graph Laplacian weights on the welded surface (inverse edge length, scale free).
    i = weld_faces[:, [0, 1, 2]].ravel()
    j = weld_faces[:, [1, 2, 0]].ravel()
    keep = i != j
    i, j = i[keep], j[keep]
    length = np.linalg.norm(weld_pos[i] - weld_pos[j], axis=1)
    mean_len = float(length.mean()) if length.size else 1.0
    w = mean_len / np.maximum(length, 1e-3 * mean_len)
    adjacency = coo_matrix((np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                           shape=(nw, nw)).tocsr()
    adjacency.sum_duplicates()
    return SurfaceTopology(f, weld, nw, weld_faces, weld_pos, face_chart, int(len(uniq)),
                           face_normal, face_area, adjacency)


# --------------------------------------------------------------------------------------
# UV rasterisation
# --------------------------------------------------------------------------------------

@dataclass
class UVRaster:
    size: int
    texel: np.ndarray      # (N,) int32 flat row-major index, row 0 = v_down 0 (image top)
    face: np.ndarray       # (N,) int32
    bary: np.ndarray       # (N,3) float32
    conservative: int = 0  # texels assigned to triangles that contained no texel centre
    overlaps: int = 0      # texel centres claimed by more than one triangle (UV overlap/shared edges)
    # Secondary claims from OTHER surfaces sharing the texel (mirrored / overlapping UV islands).
    # They are projected too and fused into the same texel, so a texel is observed when any of
    # the surfaces that use it is visible.
    extra_pos: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))    # index into texel
    extra_face: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    extra_bary: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))


def rasterize_uv(uv_vdown, faces, size, chunk=1 << 21):
    """Texel-centre rasterisation in v_down image space (texel (x,y) centre = (x+.5, y+.5)/size)."""
    size = int(size)
    if not 16 <= size <= 8192:
        raise ValueError("Atlas size must be between 16 and 8192")
    uv = np.asarray(uv_vdown, np.float64) * size
    f = np.asarray(faces, np.int64)
    a, b, c = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    i0 = np.clip(np.ceil(lo[:, 0] - 0.5), 0, size - 1).astype(np.int64)
    i1 = np.clip(np.floor(hi[:, 0] - 0.5), -1, size - 1).astype(np.int64)
    j0 = np.clip(np.ceil(lo[:, 1] - 0.5), 0, size - 1).astype(np.int64)
    j1 = np.clip(np.floor(hi[:, 1] - 0.5), -1, size - 1).astype(np.int64)
    nx = i1 - i0 + 1
    ny = j1 - j0 + 1
    good = (np.abs(area2) > 1e-12) & (nx > 0) & (ny > 0) & np.isfinite(area2)
    counts = np.where(good, nx * ny, 0).astype(np.int64)
    inv = np.where(good, 1.0 / np.where(good, area2, 1.0), 0.0)
    # affine barycentric coefficients: w = A*px + B*py + C
    A0, B0, C0 = (b[:, 1] - c[:, 1]) * inv, (c[:, 0] - b[:, 0]) * inv, (b[:, 0] * c[:, 1] - c[:, 0] * b[:, 1]) * inv
    A1, B1, C1 = (c[:, 1] - a[:, 1]) * inv, (a[:, 0] - c[:, 0]) * inv, (c[:, 0] * a[:, 1] - a[:, 0] * c[:, 1]) * inv
    cum = np.cumsum(counts)
    total = int(cum[-1]) if cum.size else 0
    out_t, out_f, out_b, out_m = [], [], [], []
    start = 0
    nfaces = len(f)
    eps = 1e-7
    while start < nfaces and total:
        base = int(cum[start - 1]) if start else 0
        stop = int(np.searchsorted(cum, base + chunk, side="right"))
        stop = max(stop, start + 1)
        cnt = counts[start:stop]
        n = int(cnt.sum())
        if n:
            fid = np.repeat(np.arange(start, stop, dtype=np.int64), cnt)
            offsets = np.repeat(np.cumsum(cnt) - cnt, cnt)
            k = np.arange(n, dtype=np.int64) - offsets
            w_x = nx[fid]
            xi = i0[fid] + k % w_x
            yi = j0[fid] + k // w_x
            px = xi + 0.5
            py = yi + 0.5
            w0 = A0[fid] * px + B0[fid] * py + C0[fid]
            w1 = A1[fid] * px + B1[fid] * py + C1[fid]
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
            if inside.any():
                out_t.append(yi[inside] * size + xi[inside])
                out_f.append(fid[inside])
                bb = np.stack((w0[inside], w1[inside], w2[inside]), 1)
                out_b.append(bb.astype(np.float32))
                out_m.append(np.min(bb, 1).astype(np.float32))
        start = stop
        _interrupt()
    if out_t:
        texel = np.concatenate(out_t)
        face = np.concatenate(out_f)
        bary = np.concatenate(out_b)
        minb = np.concatenate(out_m)
    else:
        texel = np.zeros(0, np.int64)
        face = np.zeros(0, np.int64)
        bary = np.zeros((0, 3), np.float32)
        minb = np.zeros(0, np.float32)
    order = np.argsort(texel, kind="stable")
    texel, face, bary, minb = texel[order], face[order], bary[order], minb[order]
    overlaps = 0
    extra_t = np.zeros(0, np.int64)
    extra_f = np.zeros(0, np.int64)
    extra_b = np.zeros((0, 3), np.float32)
    if texel.size > 1:
        dup = texel[1:] == texel[:-1]
        if dup.any():
            # The most interior claim owns the texel (shared edges, overlapping UVs).
            in_run = np.zeros(texel.size, bool)
            in_run[1:] |= dup
            in_run[:-1] |= dup
            idx = np.flatnonzero(in_run)
            sub = np.lexsort((-minb[idx], texel[idx]))
            idx = idx[sub]
            first = np.ones(idx.size, bool)
            first[1:] = texel[idx[1:]] != texel[idx[:-1]]
            drop = idx[~first]
            owner = idx[np.maximum.accumulate(np.where(first, np.arange(idx.size), 0))][~first]
            # Secondary claims of a neighbouring triangle (shared edge) are the same surface point;
            # claims of an unrelated triangle (mirrored / overlapping island) are kept as extras.
            shared = (f[face[drop]][:, :, None] == f[face[owner]][:, None, :]).any(2).sum(1)
            other = shared < 2
            extra_t, extra_f, extra_b = texel[drop[other]], face[drop[other]], bary[drop[other]]
            keep = np.ones(texel.size, bool)
            keep[drop] = False
            overlaps = int(drop.size)
            texel, face, bary = texel[keep], face[keep], bary[keep]
    # Conservative coverage: triangles without any texel centre get the texel under their
    # UV centroid when it is still free (thin/tiny islands otherwise vanish or bleed).
    hit = np.zeros(nfaces, bool)
    hit[face] = True
    missing = np.flatnonzero(~hit)
    added = 0
    if missing.size:
        cen = (a[missing] + b[missing] + c[missing]) / 3.0
        ok = np.isfinite(cen).all(1)
        missing, cen = missing[ok], cen[ok]
        xi = np.clip(np.floor(cen[:, 0]), 0, size - 1).astype(np.int64)
        yi = np.clip(np.floor(cen[:, 1]), 0, size - 1).astype(np.int64)
        t = yi * size + xi
        pos = np.searchsorted(texel, t)
        pos = np.minimum(pos, max(texel.size - 1, 0))
        taken = (texel.size > 0) & (texel[pos] == t) if texel.size else np.zeros(t.size, bool)
        t, missing = t[~taken], missing[~taken]
        t, first = np.unique(t, return_index=True)
        missing = missing[first]
        if t.size:
            added = int(t.size)
            texel = np.concatenate([texel, t])
            face = np.concatenate([face, missing])
            bary = np.concatenate([bary, np.full((t.size, 3), 1.0 / 3.0, np.float32)])
            order = np.argsort(texel, kind="stable")
            texel, face, bary = texel[order], face[order], bary[order]
    extra_pos = np.searchsorted(texel, extra_t).astype(np.int32)
    return UVRaster(size, texel.astype(np.int32), face.astype(np.int32), np.ascontiguousarray(bary, np.float32),
                    added, overlaps, extra_pos, extra_f.astype(np.int32), np.ascontiguousarray(extra_b, np.float32))


# --------------------------------------------------------------------------------------
# cached surface preparation
# --------------------------------------------------------------------------------------

@dataclass
class Surface:
    key: tuple
    topology: SurfaceTopology
    raster: UVRaster
    texel_chart: np.ndarray    # (N,) int32
    texel_area: np.ndarray     # (N,) float32 3D surface area represented by each texel
    extra_area: np.ndarray     # (E,) float32 same for the secondary (overlapping-UV) samples
    vertices: np.ndarray       # normalised positions (float32)
    normals: np.ndarray        # conditioned shading normals (float32)
    timings: dict = field(default_factory=dict)
    _gutter_source: object = None   # (T*T,) int32 index of the nearest occupied texel (lazy)

    def gutter_source(self):
        """For every atlas pixel, the flat index of the nearest occupied texel (EDT, cached)."""
        if self._gutter_source is None:
            from scipy.ndimage import distance_transform_edt
            T = self.raster.size
            covered = np.zeros(T * T, bool)
            covered[self.raster.texel] = True
            covered = covered.reshape(T, T)
            if covered.all():
                src = np.arange(T * T, dtype=np.int32)
            else:
                iy, ix = distance_transform_edt(~covered, return_distances=False, return_indices=True)
                src = (iy.astype(np.int32) * T + ix.astype(np.int32)).reshape(-1)
                del iy, ix
            self._gutter_source = src
        return self._gutter_source


_SURFACE_CACHE: "OrderedDict[tuple, Surface]" = OrderedDict()
_SURFACE_CACHE_LIMIT = 2


def uv_vdown(context):
    uv = np.asarray(context.uvs, np.float64)
    if getattr(context, "uv_convention", "v_down") == "v_up":
        uv = uv.copy()
        uv[:, 1] = 1.0 - uv[:, 1]
    return uv


def prepare_surface(context, size):
    if context.uvs is None:
        raise ValueError("This mesh has no UVs. Run 'UTX 0. UV Unwrap Mesh' before baking.")
    uv = np.asarray(context.uvs, np.float64)
    if not np.isfinite(uv).all():
        raise ValueError("UVs contain NaN/inf values")
    if uv.min() < -1e-4 or uv.max() > 1 + 1e-4:
        raise ValueError("UVs outside [0,1] (UDIM / tiled / offset layouts) cannot be baked into one atlas. "
                         "Re-unwrap the mesh with 'UTX 0. UV Unwrap Mesh' (comfy_gpu or xatlas).")
    key = (str(context.atlas_id), str(context.geometry_id), int(size))
    cached = _SURFACE_CACHE.get(key)
    if cached is not None:
        _SURFACE_CACHE.move_to_end(key)
        return cached, True
    t0 = time.perf_counter()
    topology = build_topology(context.vertices, context.faces, context.uvs)
    t1 = time.perf_counter()
    raster = rasterize_uv(uv_vdown(context), context.faces, size)
    t2 = time.perf_counter()
    if raster.texel.size == 0:
        raise ValueError("The UV layout covers no texel centre at this texture size")
    texel_chart = topology.face_chart[raster.face]
    nf = len(topology.faces)
    per_face = (np.bincount(raster.face, minlength=nf) + np.bincount(raster.extra_face, minlength=nf)).astype(np.float64)
    texel_area = (topology.face_area[raster.face] / np.maximum(per_face[raster.face], 1)).astype(np.float32)
    extra_area = (topology.face_area[raster.extra_face] / np.maximum(per_face[raster.extra_face], 1)).astype(np.float32)
    surface = Surface(key, topology, raster, texel_chart.astype(np.int32), texel_area, extra_area,
                      np.array(context.vertices, np.float32), np.array(context.normals, np.float32),
                      {"topology": round(t1 - t0, 4), "uv_raster": round(t2 - t1, 4)})
    _SURFACE_CACHE[key] = surface
    while len(_SURFACE_CACHE) > _SURFACE_CACHE_LIMIT:
        _SURFACE_CACHE.popitem(last=False)
    return surface, False


def clear_surface_cache():
    _SURFACE_CACHE.clear()


# --------------------------------------------------------------------------------------
# torch side: per-texel geometry, views, projection
# --------------------------------------------------------------------------------------

def _torch():
    import torch
    return torch


@dataclass
class TexelGeometry:
    P: object    # (N+E,3) normalised 3D point (primary texels first, then overlapping-UV extras)
    Ns: object   # (N+E,3) unit shading normal
    Ng: object   # (N+E,3) unit geometric (plane) normal
    A: object    # (N+E,) 3D area represented by the sample
    n_primary: int = -1
    extra_pos: object = None   # (E,) texel index of each extra sample

    def slice(self, s, e):
        return TexelGeometry(self.P[s:e], self.Ns[s:e], self.Ng[s:e], self.A[s:e], e - s)

    def index(self, idx):
        return TexelGeometry(self.P[idx], self.Ns[idx], self.Ng[idx], self.A[idx], int(idx.numel()))


def texel_geometry(surface, device, chunk=1 << 20):
    """Per-texel 3D point, unit shading normal, unit geometric normal and area on `device`."""
    torch = _torch()
    r = surface.raster
    n_primary = r.texel.size
    face_all = np.concatenate([r.face, r.extra_face]) if r.extra_face.size else r.face
    bary_all = np.concatenate([r.bary, r.extra_bary]) if r.extra_face.size else r.bary
    n = face_all.size
    V = torch.from_numpy(surface.vertices).to(device)
    Nv = torch.from_numpy(surface.normals).to(device)
    faces = torch.from_numpy(surface.topology.faces).to(device)
    fnorm = torch.from_numpy(surface.topology.face_normal).to(device)
    P = torch.empty((n, 3), dtype=torch.float32, device=device)
    Ns = torch.empty_like(P)
    Ng = torch.empty_like(P)
    area = np.concatenate([surface.texel_area, surface.extra_area]) if r.extra_face.size else surface.texel_area
    A = torch.from_numpy(np.ascontiguousarray(area)).to(device)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        fid = torch.from_numpy(face_all[s:e]).to(device).long()
        bary = torch.from_numpy(bary_all[s:e]).to(device)
        tri = faces[fid]
        P[s:e] = (V[tri[:, 0]] * bary[:, :1] + V[tri[:, 1]] * bary[:, 1:2] + V[tri[:, 2]] * bary[:, 2:3])
        ns = (Nv[tri[:, 0]] * bary[:, :1] + Nv[tri[:, 1]] * bary[:, 1:2] + Nv[tri[:, 2]] * bary[:, 2:3])
        g = fnorm[fid]
        length = ns.norm(dim=1, keepdim=True)
        # A vanishing interpolated normal (opposing vertex normals) falls back to the face plane,
        # oriented like the vertex normals.
        sign = torch.where((g * ns).sum(1, keepdim=True) < 0, -1.0, 1.0)
        Ns[s:e] = torch.where(length > 1e-6, ns / length.clamp_min(1e-12), g * sign)
        Ng[s:e] = g
    extra = torch.from_numpy(r.extra_pos.astype(np.int64)).to(device) if r.extra_face.size else None
    return TexelGeometry(P, Ns, Ng, A, n_primary, extra)


@dataclass
class ViewInput:
    """One calibrated generated view (numpy, lazily loaded)."""
    label: str
    matrix: np.ndarray          # (4,4) world(normalised mesh) -> camera
    image_ref: object           # ArrayRef (S,S,3) uint8
    mask_ref: object            # ArrayRef (S,S)
    depth_ref: object           # ArrayRef (R,R) float32 linear depth, 0 = background


def collect_views(collection):
    views = []
    for si, vs in enumerate(collection.sets):
        for vi in getattr(vs.plan, "active_indices", tuple(range(len(vs.images)))):
            # 10-view generation keeps a six-tile DiT layout internally but
            # only four of the diagonal rig's tiles contain actual images.
            views.append(ViewInput(f"rig{si}:{vs.plan.names[vi] if vi < len(vs.plan.names) else vi}",
                                   np.asarray(vs.plan.matrices[vi], np.float64), vs.images[vi], vs.masks[vi],
                                   vs.gbuffer.depths[vi]))
    return views


@dataclass
class PreparedView:
    label: str
    rot: object
    trans: object
    center: object
    depth: object      # (R*R,) float32
    R: int
    levels: list       # [(1,C+1,S,S), (1,C+1,S/2,S/2), (1,C+1,S/4,S/4)] premultiplied by validity
    S: int
    edge_w: object     # (S*S,) float32 edge distance weight
    fg_fraction: float
    detail: object = None   # ViewDetail (detail mode only)


@dataclass
class ViewDetail:
    """Extra per-view data for the detail mode (Base Color only)."""
    fine_base: object  # (1, C, S, S): mask-normalised G(DETAIL_SIGMA_FINE) of the view, linear
    win_ok: object     # (S-3, S-3) bool: the 4x4 cubic window starting at (row, col) is fully valid
    lo: object         # (C, (S-1)*(S-1)): min of each 2x2 pixel block (anti-ringing clamp)
    hi: object         # (C, (S-1)*(S-1)): max of each 2x2 pixel block


def _gauss_kernel(sigma, torch, device):
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum(), r


def blur_premultiplied(image_t, sigma):
    """Mask-normalised separable Gaussian of a premultiplied (1, C+1, S, S) image whose last channel
    is the validity. Returns (1, C, S, S); invalid pixels never leak colour into valid ones.
    Shift-and-add instead of a grouped conv2d: the same result with about a third of the temporary
    memory (2048 view on CPU: 192 instead of 641 MiB peak, and 3x faster)."""
    import torch
    import torch.nn.functional as F
    k, r = _gauss_kernel(sigma, torch, image_t.device)
    S = image_t.shape[-1]
    mode = "reflect" if r < S else "replicate"
    w = k.tolist()

    def separable_pass(x, dim):
        n = x.shape[dim]
        xp = F.pad(x, (r, r, 0, 0) if dim == 3 else (0, 0, r, r), mode=mode)
        out = xp.narrow(dim, 0, n) * w[0]
        for i in range(1, 2 * r + 1):
            out.add_(xp.narrow(dim, i, n), alpha=w[i])
        return out

    x = separable_pass(separable_pass(image_t, 3), 2)
    return x[:, :-1] / x[:, -1:].clamp_min(1e-4)


def _view_detail(image_t, torch):
    """Fine-band base image and the cubic-window validity map for one prepared Base Color view.
    Depth-edge pixels are already invalid in image_t, so the narrow (sigma 1 px) mask-normalised
    blur does not carry colour across occlusion boundaries. Built in the order that keeps the
    fewest temporaries alive (the detail mode's memory peak is here)."""
    import torch.nn.functional as F
    full = (image_t[:, -1:] > 0.999).to(image_t.dtype)
    win_ok = F.max_pool2d(full.neg_(), 4, stride=1).neg_()[0, 0] > 0.5
    del full
    fine_base = blur_premultiplied(image_t, DETAIL_SIGMA_FINE).contiguous()
    data = image_t[:, :-1]
    C = data.shape[1]
    hi = F.max_pool2d(data, 2, stride=1)[0].reshape(C, -1).contiguous()
    lo = F.max_pool2d(data.neg(), 2, stride=1).neg_()[0].reshape(C, -1).contiguous()
    return ViewDetail(fine_base, win_ok, lo, hi)


def _edge_weight(depth_t, S, torch):
    """(weight (S*S,) in [EDGE_FLOOR,1] from the distance in view pixels to silhouettes/depth
    jumps, edge-pixel mask (S,S)). Edge pixels straddle two surfaces (or surface and background):
    their generated colour is a mixture, so they are also excluded from colour sampling."""
    import torch.nn.functional as F
    R = depth_t.shape[0]
    d = depth_t
    fg = d > 0
    pix = d / (FOCAL * R)
    edges = torch.zeros_like(fg)
    # horizontal / vertical neighbour jumps; threshold scales with the local pixel footprint
    for dim in (0, 1):
        a = d.narrow(dim, 0, R - 1)
        b = d.narrow(dim, 1, R - 1)
        fa = fg.narrow(dim, 0, R - 1)
        fb = fg.narrow(dim, 1, R - 1)
        thr = 1e-3 + 12.0 * torch.maximum(pix.narrow(dim, 0, R - 1), pix.narrow(dim, 1, R - 1))
        jump = (fa != fb) | (fa & fb & ((a - b).abs() > thr))
        e_a = edges.narrow(dim, 0, R - 1)
        e_b = edges.narrow(dim, 1, R - 1)
        e_a |= jump
        e_b |= jump
    e = edges.float()[None, None]
    if R != S:
        if R % S == 0:
            e = F.max_pool2d(e, R // S)
        else:
            e = (F.interpolate(e, size=(S, S), mode="area") > 0).float()
    ramp = max(2.0, 3.0 * S / 1024.0)
    steps = int(math.ceil(ramp))
    dist = torch.full((S, S), float(steps), device=depth_t.device)
    cur = e
    dist[cur[0, 0] > 0] = 0.0
    for k in range(1, steps):
        cur = F.max_pool2d(cur, 3, 1, 1)
        newly = (cur[0, 0] > 0) & (dist > k)
        dist[newly] = float(k)
    t = (dist / ramp).clamp(0, 1)
    w = EDGE_FLOOR + (1 - EDGE_FLOOR) * (t * t * (3 - 2 * t))
    return w.reshape(-1).contiguous(), dist == 0


def prepare_view(view: ViewInput, semantic, device, detail=False):
    """detail=True also attaches the detail-mode data (Base Color only); fuse_views attaches it
    lazily per view instead, so the gains pass and the view cache stay as light as before."""
    torch = _torch()
    try:
        depth = np.asarray(view.depth_ref.read(), np.float32)
        image = np.asarray(view.image_ref.read())
        mask = np.asarray(view.mask_ref.read())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"{view.label}: generated view or its G-buffer depth is missing from the UltraTex "
                           f"cache ({error}). The cache folder was probably cleared after generation: re-run "
                           f"the Generate node (or load a saved view bundle).") from error
    if depth.ndim != 2 or depth.shape[0] != depth.shape[1]:
        raise ValueError(f"{view.label}: depth buffer must be square")
    if image.ndim != 3 or image.shape[0] != image.shape[1] or image.shape[2] < 3:
        raise ValueError(f"{view.label}: generated view must be a square RGB image")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"{view.label}: view mask and image sizes differ")
    S = image.shape[0]
    R = depth.shape[0]
    rgb = image[..., :3].astype(np.float32) / (255.0 if image.dtype == np.uint8 else 1.0)
    valid = (np.asarray(mask, np.float32) > 0.5).astype(np.float32)
    if semantic == "base_color":
        data = srgb_to_linear_np(rgb)
    elif semantic == "mr":
        data = rgb[..., 1:3]  # UltraTex MR: G = roughness, B = metallic (linear data)
    else:
        raise ValueError(f"Unsupported bake semantic {semantic}")
    depth_t = torch.from_numpy(np.array(depth, np.float32)).to(device)
    edge_w, edge_px = _edge_weight(depth_t, S, torch)
    valid_t = torch.from_numpy(valid).to(device) * (~edge_px).to(torch.float32)
    data_t = torch.from_numpy(np.ascontiguousarray(data.transpose(2, 0, 1))).to(device)
    image_t = torch.cat([data_t * valid_t[None], valid_t[None]], 0)[None].contiguous()
    # Mask-normalised mip levels: used when one atlas texel covers several view pixels.
    import torch.nn.functional as F
    levels = [image_t]
    for _ in range(2):
        if levels[-1].shape[-1] % 2 or levels[-1].shape[-1] < 32:
            break
        levels.append(F.avg_pool2d(levels[-1], 2))
    M = np.asarray(view.matrix, np.float64)
    rot = M[:3, :3]
    trans = M[:3, 3]
    center = np.linalg.inv(M)[:3, 3]
    pv = PreparedView(view.label,
                      torch.tensor(rot, dtype=torch.float32, device=device),
                      torch.tensor(trans, dtype=torch.float32, device=device),
                      torch.tensor(center, dtype=torch.float32, device=device),
                      depth_t.reshape(-1).contiguous(), R, levels, S, edge_w,
                      float(valid.mean()))
    if detail and semantic == "base_color":
        attach_detail(pv)
    return pv


def attach_detail(pv):
    """Attach the detail-mode data (fine-band base image, cubic-window map) to a prepared
    Base Color view. Views smaller than 16 px keep the standard sampling."""
    if pv.detail is None and pv.S >= 16 and pv.levels[0].shape[1] == 4:
        pv.detail = _view_detail(pv.levels[0], _torch())
    return pv


def _sample_levels(view, u, v, lod, torch):
    """Trilinear (mip) mask-normalised sample. lod in view-pixel log2 footprint units.
    Coarser levels are fetched only for texels whose footprint exceeds ~1 view pixel."""
    import torch.nn.functional as F

    def fetch(level, uu, vv):
        grid = torch.stack((2 * uu - 1, 2 * vv - 1), -1).clamp(-1.5, 1.5).view(1, 1, -1, 2)
        s = F.grid_sample(view.levels[level], grid, mode="bilinear", padding_mode="border", align_corners=False)
        return s[0, :, 0, :].T

    out = fetch(0, u, v)
    top = len(view.levels) - 1
    if top >= 1:
        need = (lod > 0.05).nonzero(as_tuple=True)[0]
        if need.numel():
            l = lod[need].clamp(0, top)
            uu, vv = u[need], v[need]
            w0 = (1 - l).clamp(0, 1)
            w2 = (l - 1).clamp(0, 1) if top >= 2 else torch.zeros_like(l)
            w1 = 1 - w0 - w2
            mixed = out[need] * w0[:, None] + fetch(1, uu, vv) * w1[:, None]
            if top >= 2 and bool((w2 > 0).any()):
                mixed = mixed + fetch(2, uu, vv) * w2[:, None]
            out[need] = mixed
    return out


def _sample_cubic(view, u, v, x0, y0, colour, lod, torch, need=None):
    """Detail mode: replace `colour` (bilinear, mask-normalised) by a cubic (Keys a=-0.75)
    reconstruction wherever the texel magnifies the view (lod ~ 0) and the 4x4 window is fully
    valid; the result is clamped to the 2x2 nearest view pixels (no ringing / halos).
    `need` (bool) limits the work to samples whose fine band can still matter."""
    import torch.nn.functional as F
    d = view.detail
    S = view.S
    n_ok = d.win_ok.shape[0]
    wi = y0 - 1
    wj = x0 - 1
    cand = (wi >= 0) & (wj >= 0) & (wi < n_ok) & (wj < n_ok) & (lod <= DETAIL_LOD_MAX)
    if need is not None:
        cand &= need
    ok = torch.zeros_like(cand)
    ci = cand.nonzero(as_tuple=True)[0]
    if ci.numel() == 0:
        return colour
    ok[ci] = d.win_ok[wi[ci], wj[ci]]
    sel = ok.nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        return colour
    C = colour.shape[1]
    lvl = view.levels[0]
    grid = torch.stack((2 * u[sel] - 1, 2 * v[sel] - 1), -1).view(1, 1, -1, 2)
    cub = F.grid_sample(lvl[:, :C], grid, mode="bicubic", padding_mode="border", align_corners=False)[0, :, 0, :].T
    blk = y0[sel] * (S - 1) + x0[sel]
    colour[sel] = torch.maximum(torch.minimum(cub, d.hi[:, blk].T), d.lo[:, blk].T)
    return colour


def _project(geo, view: PreparedView, torch, detail=False, with_footprint=False, qfloor=None):
    """Return (colour (n,C), quality (n,)) for one view; quality 0 = not observed.

    detail=True (view prepared with detail data): returns (colour, quality, fine_base (n, C)) with
    cubic reconstruction and the view's G(DETAIL_SIGMA_FINE) image at the same position.
    with_footprint=True: returns (colour, quality, footprint (n,)) = view pixels per texel side.
    qfloor (n,): detail mode only, skip the cubic reconstruction where q < qfloor (the fine band
    of such a sample is outweighed by a better camera already fused).

    Back-facing / out-of-frame texels are culled first, so the depth tests and image
    fetches only run on the candidates (about half of the texels per view)."""
    P, Ns = geo.P, geo.Ns
    n = P.shape[0]
    Pc = P @ view.rot.T + view.trans
    zd = -Pc[:, 2]
    inv = 1.0 / zd.clamp_min(1e-6)
    u = 0.5 + FOCAL * Pc[:, 0] * inv
    v = 0.5 - FOCAL * Pc[:, 1] * inv
    to_cam = view.center - P
    cos = (Ns * to_cam).sum(1) / to_cam.norm(dim=1).clamp_min(1e-9)
    ok = (zd > NEAR) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1) & (cos > COS_MIN)
    C = view.levels[0].shape[1] - 1
    detail = bool(detail and view.detail is not None)
    colour_out = torch.zeros((n, C), dtype=torch.float32, device=P.device)
    q_out = torch.zeros(n, dtype=torch.float32, device=P.device)
    extra_out = None
    if detail:
        extra_out = torch.zeros((n, C), dtype=torch.float32, device=P.device)
    elif with_footprint:
        extra_out = torch.zeros(n, dtype=torch.float32, device=P.device)
    idx = ok.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return (colour_out, q_out) if extra_out is None else (colour_out, q_out, extra_out)
    Pc, zd, inv, u, v, cos = Pc[idx], zd[idx], inv[idx], u[idx], v[idx], cos[idx]
    Ng = geo.Ng[idx]
    A = geo.A[idx]
    R = view.R
    x = u * R - 0.5
    y = v * R - 0.5
    x0 = x.floor()
    y0 = y.floor()
    fx = x - x0
    fy = y - y0
    nc = Ng @ view.rot.T
    num = (nc * Pc).sum(1)
    tol = TOL_ABS + TOL_PX * zd / (FOCAL * R)
    support = torch.zeros_like(zd)
    front = torch.zeros_like(zd)
    near_same = torch.zeros(zd.shape, dtype=torch.bool, device=zd.device)
    nx_sel = fx >= 0.5
    ny_sel = fy >= 0.5
    for dx in (0, 1):
        xi = (x0 + dx).clamp(0, R - 1)
        xn = ((xi + 0.5) / R - 0.5) / FOCAL
        wx = fx if dx else 1 - fx
        for dy in (0, 1):
            yi = (y0 + dy).clamp(0, R - 1)
            yn = (0.5 - (yi + 0.5) / R) / FOCAL
            D = view.depth[yi.long() * R + xi.long()]
            # depth the texel's own triangle plane has on this pixel's ray
            den = nc[:, 0] * xn + nc[:, 1] * yn - nc[:, 2]
            steep = den.abs() > 2e-2
            t = torch.where(steep, num / torch.where(steep, den, torch.ones_like(den)), zd)
            fg = D > 0
            same = fg & ((D - t).abs() <= tol)
            fr = fg & (D < t - tol)
            bw = wx * (fy if dy else 1 - fy)
            support += bw * same
            front += bw * fr
            sel = (nx_sel == bool(dx)) & (ny_sel == bool(dy))
            near_same |= sel & same
    vis = torch.where(near_same, support * (1 - front) ** 2, torch.zeros_like(support))
    # footprint of the texel in view pixels (geometric mean of the projected axes)
    footprint = (A * cos.clamp_min(0.05)).clamp_min(0).sqrt() * (view.S * FOCAL) * inv
    lod = torch.log2(footprint.clamp_min(1.0))
    sample = _sample_levels(view, u, v, lod, torch)
    alpha = sample[:, -1]
    colour = sample[:, :-1] / alpha.clamp_min(1e-4)[:, None]
    S = view.S
    xs = (u * S).floor().clamp(0, S - 1).long()
    ys = (v * S).floor().clamp(0, S - 1).long()
    edge = view.edge_w[ys * S + xs]
    q = torch.where(alpha > 0.05, cos.clamp(0, 1) * vis * edge, torch.zeros_like(vis))
    if detail:
        import torch.nn.functional as F
        x0 = (u * S - 0.5).floor().long()
        y0 = (v * S - 0.5).floor().long()
        need = q >= Q_LO if qfloor is None else (q >= Q_LO) & (q >= qfloor[idx])
        colour = _sample_cubic(view, u, v, x0, y0, colour, lod, torch, need)
        grid = torch.stack((2 * u - 1, 2 * v - 1), -1).view(1, 1, -1, 2)
        extra_out[idx] = F.grid_sample(view.detail.fine_base, grid, mode="bilinear", padding_mode="border",
                                       align_corners=False)[0, :, 0, :].T
    elif with_footprint:
        extra_out[idx] = footprint
    colour_out[idx] = colour
    q_out[idx] = q
    return (colour_out, q_out) if extra_out is None else (colour_out, q_out, extra_out)


def _smoothstep(x, lo, hi):
    t = np.clip((np.asarray(x, np.float32) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _as_prepared(view, semantic, device):
    return view if isinstance(view, PreparedView) else prepare_view(view, semantic, device)


def _view_bytes(view, semantic):
    R = int(view.depth_ref.shape[0])
    S = int(view.image_ref.shape[0])
    C = 3 if semantic == "base_color" else 2
    return R * R * 4 + int(S * S * (C + 1) * 4 * 1.34) + S * S * 4


def _density_report(Q, FP):
    """How finely the best camera samples the atlas: view pixels per texel side (median etc.).
    Below 1 the atlas is finer than the views (magnified): detail above the views' own
    resolution cannot come from the views."""
    if Q.size == 0:
        return {}
    best = Q.argmax(0)
    qb = Q.max(0)
    fp = FP[best, np.arange(Q.shape[1])]
    good = qb >= Q_HI
    if not good.any():
        return {}
    fp = fp[good]
    med = float(np.median(fp))
    return {"view_px_per_texel_median": round(med, 3),
            "view_px_per_texel_p10": round(float(np.percentile(fp, 10)), 3),
            "view_px_per_texel_p90": round(float(np.percentile(fp, 90)), 3),
            "texels_magnified_fraction": round(float((fp < 0.95).mean()), 3),
            "note": ("the atlas is %.1fx finer (linear) than the best camera samples the surface: detail "
                     "finer than the generated views cannot be recovered by baking" % (1.0 / max(med, 1e-6)))
                    if med < 0.95 else "the atlas does not exceed the views' sampling density"}


def estimate_gains(geo, views, device, max_samples=200_000):
    """Robust per-view, per-channel multiplicative colour gains (log-domain least squares).
    Also returns the atlas/view sampling-density statistics of the subsample."""
    torch = _torch()
    n = geo.n_primary if geo.n_primary >= 0 else geo.P.shape[0]
    count = len(views)
    if count < 2 or n == 0:
        return np.ones((count, 3), np.float32), {"pairs_used": 0}
    step = max(1, n // max_samples)
    idx = torch.arange(0, n, step, device=geo.P.device)
    sub = geo.index(idx)
    cols, qs, fps = [], [], []
    for view in views:
        _interrupt()
        pv = _as_prepared(view, "base_color", device)
        c, q, fp = _project(sub, pv, torch, with_footprint=True)
        cols.append(c.float().cpu().numpy())
        qs.append(q.float().cpu().numpy())
        fps.append(fp.float().cpu().numpy())
        del pv
    C = np.stack(cols)  # (V,n,3)
    Q = np.stack(qs)    # (V,n)
    density = _density_report(Q, np.stack(fps))
    logC = np.log(np.clip(C, 1e-3, None))
    lum_ok = C.mean(2) > 0.01
    gains = np.zeros((count, 3))
    pairs = 0
    for ch in range(3):
        A = np.zeros((count, count))
        bvec = np.zeros(count)
        pair_terms = []
        for i in range(count):
            for j in range(i + 1, count):
                w = np.minimum(Q[i], Q[j])
                m = (w > 0.2) & lum_ok[i] & lum_ok[j]
                if m.sum() < 64:
                    continue
                d = (logC[i, m, ch] - logC[j, m, ch])
                pair_terms.append((i, j, w[m], d))
        if not pair_terms:
            continue
        pairs = max(pairs, len(pair_terms))
        x = np.zeros(count)
        for _ in range(3):  # IRLS with a Cauchy-like robust weight
            A[:] = 0
            bvec[:] = 0
            for i, j, w, d in pair_terms:
                r = x[i] - x[j] + d
                rw = w / (1.0 + (r / 0.1) ** 2)
                s = rw.sum()
                sd = (rw * d).sum()
                A[i, i] += s
                A[j, j] += s
                A[i, j] -= s
                A[j, i] -= s
                bvec[i] -= sd
                bvec[j] += sd
            lam = 0.05 * max(np.trace(A) / count, 1e-6)
            x = np.linalg.solve(A + lam * np.eye(count), bvec)
            x -= x.mean()
        gains[:, ch] = x
    g = np.exp(np.clip(gains, math.log(1 - GAIN_LIMIT), math.log(1 + GAIN_LIMIT))).astype(np.float32)
    return g, {"pairs_used": int(pairs), "samples": int(idx.numel()), "sampling_density": density}


def fuse_views(surface, geo, views, semantic, device, gains=None, chunk=1 << 20, detail=False):
    """Two-band confidence-weighted fusion.

    sharp: weights q**p_sharp (near best-view, keeps detail crisp, no hard switch)
    broad: weights q**FUSION_POWER_BROAD (wide smooth hand-over between cameras)
    The caller keeps the high frequencies of `sharp` and the low frequencies of `broad`.

    detail=True (Base Color): each view sample is split into its fine band
    H = cubic sample - G(DETAIL_SIGMA_FINE) and the rest. The rest goes through exactly the
    sharp/broad fusion above; H is fused separately with weights (q/Q_REF)**DETAIL_POWER_FINE
    (finest detail from ~one camera, narrow hand-over). A 7th value (fused H, best camera's full
    sharp sample) is returned; bake_branch adds H and bounds the result (see DETAIL_RANGE_SLACK)."""
    torch = _torch()
    n = geo.n_primary if geo.n_primary >= 0 else geo.P.shape[0]
    n_all = geo.P.shape[0]
    C = 3 if semantic == "base_color" else 2
    detail = bool(detail and semantic == "base_color")
    p_sharp = FUSION_POWER if len(views) <= 6 else FUSION_POWER_MANY
    acc_s = torch.zeros((n, C), dtype=torch.float32, device=device)
    acc_b = torch.zeros((n, C), dtype=torch.float32, device=device)
    w_s = torch.zeros(n, dtype=torch.float32, device=device)
    w_b = torch.zeros(n, dtype=torch.float32, device=device)
    qmax = torch.zeros(n, dtype=torch.float32, device=device)
    best = torch.zeros((n, C), dtype=torch.float32, device=device)
    nviews = torch.zeros(n, dtype=torch.int16, device=device)
    acc_h = torch.zeros((n, C), dtype=torch.float32, device=device) if detail else None
    w_h = torch.zeros(n, dtype=torch.float32, device=device) if detail else None
    best_sharp = torch.zeros((n, C), dtype=torch.float32, device=device) if detail else None
    inv_ref = 1.0 / DETAIL_Q_REF
    per_view = []

    def split(res, g):
        """(base fused by the standard bands, fine detail H or None, q, full sharp sample or None)."""
        colour, q = res[0], res[1]
        if g is not None:
            colour = colour * g
        if len(res) == 3:
            base = res[2] * g if g is not None else res[2]
            return base, colour - base, q, colour
        return colour, None, q, None

    for vi, view in enumerate(views):
        _interrupt()
        pv = _as_prepared(view, semantic, device)
        cached = pv is view
        if detail:
            attach_detail(pv)
        g = None if gains is None else torch.tensor(gains[vi][:C], dtype=torch.float32, device=device)
        used = 0
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            qfloor = DETAIL_SKIP_RATIO * qmax[s:e] if detail else None
            colour, fine, q, full = split(_project(geo.slice(s, e), pv, torch, detail=detail, qfloor=qfloor), g)
            ws = q.pow(p_sharp)
            wb = q.pow(FUSION_POWER_BROAD)
            acc_s[s:e] += colour * ws[:, None]
            acc_b[s:e] += colour * wb[:, None]
            w_s[s:e] += ws
            w_b[s:e] += wb
            if fine is not None:
                wh = (q * inv_ref).pow(DETAIL_POWER_FINE)
                acc_h[s:e] += fine * wh[:, None]
                w_h[s:e] += wh
            better = q > qmax[s:e]
            best[s:e] = torch.where(better[:, None], colour, best[s:e])
            if full is not None:
                best_sharp[s:e] = torch.where(better[:, None], full, best_sharp[s:e])
            qmax[s:e] = torch.maximum(qmax[s:e], q)
            good = q >= Q_LO
            nviews[s:e] += good.to(torch.int16)
            used += int(good.sum())
        for s in range(n, n_all, chunk):
            # overlapping-UV samples: fused into the texel they share
            e = min(n_all, s + chunk)
            colour, fine, q, full = split(_project(geo.slice(s, e), pv, torch, detail=detail), g)
            tix = geo.extra_pos[s - n:e - n]
            ws = q.pow(p_sharp)
            wb = q.pow(FUSION_POWER_BROAD)
            acc_s.index_add_(0, tix, colour * ws[:, None])
            acc_b.index_add_(0, tix, colour * wb[:, None])
            w_s.index_add_(0, tix, ws)
            w_b.index_add_(0, tix, wb)
            if fine is not None:
                wh = (q * inv_ref).pow(DETAIL_POWER_FINE)
                acc_h.index_add_(0, tix, fine * wh[:, None])
                w_h.index_add_(0, tix, wh)
            old = qmax[tix]
            qmax.scatter_reduce_(0, tix, q, reduce="amax")
            better = (q > old) & (q >= qmax[tix])
            best[tix[better]] = colour[better]
            if full is not None:
                best_sharp[tix[better]] = full[better]
        entry = {"view": pv.label, "texels_used": used, "depth_px": pv.R, "image_px": pv.S}
        if detail:
            entry["detail_bands"] = pv.detail is not None
            if cached:
                pv.detail = None     # free the band image of a cached view as soon as it is fused
        per_view.append(entry)
        del pv
    # Weights can underflow for very steep powers; the best single view is the exact limit.
    sharp = torch.where((w_s > 1e-30)[:, None], acc_s / w_s.clamp_min(1e-30)[:, None], best)
    broad = torch.where((w_b > 1e-30)[:, None], acc_b / w_b.clamp_min(1e-30)[:, None], best)
    if detail:
        # in place (same values as a where/divide): avoids three temporary (n, C) arrays at the peak
        fine = acc_h.div_(w_h.clamp_min(1e-30)[:, None]).masked_fill_(~(w_h > 1e-30)[:, None], 0.0)
        return sharp, broad, qmax, nviews, per_view, p_sharp, (fine, best_sharp)
    return sharp, broad, qmax, nviews, per_view, p_sharp


SPLAT_TEXELS = 1_500_000   # texels used to estimate per-vertex values (strided subset above this)


def _splat_subset(n):
    stride = max(1, int(math.ceil(n / SPLAT_TEXELS)))
    return slice(None, None, stride)


def _interp_vertices(surface, vert, idx=None):
    """Barycentric interpolation of per-welded-vertex values at texels (all or `idx`)."""
    r = surface.raster
    face = r.face if idx is None else r.face[idx]
    bary = r.bary if idx is None else r.bary[idx]
    wf = surface.topology.weld_faces[face]
    vert = np.asarray(vert, np.float32)
    return (vert[wf[:, 0]] * bary[:, :1] + vert[wf[:, 1]] * bary[:, 1:2] + vert[wf[:, 2]] * bary[:, 2:3])


def _splat(surface, values, weights):
    """Weighted per-vertex sums: (num (Vw,C), den (Vw,), mass (Vw,)) from a texel subset."""
    topo = surface.topology
    r = surface.raster
    nw = topo.n_weld
    values = np.asarray(values)
    weights = np.asarray(weights, np.float64)
    sub = _splat_subset(r.texel.size + r.extra_face.size)
    if r.extra_face.size:
        # overlapping-UV extras carry their texel's value to their own surface's vertices
        face = np.concatenate([r.face, r.extra_face])[sub]
        bary = np.concatenate([r.bary, r.extra_bary])[sub].astype(np.float64)
        values = np.concatenate([values, values[r.extra_pos]])[sub]
        weights = np.concatenate([weights, weights[r.extra_pos]])[sub]
    else:
        face = r.face[sub]
        bary = r.bary[sub].astype(np.float64)
        values = values[sub]
        weights = weights[sub]
    C = values.shape[1]
    wf = topo.weld_faces[face]
    num = np.zeros((nw, C))
    den = np.zeros(nw)
    mass = np.zeros(nw)
    for k in range(3):
        idx = wf[:, k]
        wb = bary[:, k]
        mass += np.bincount(idx, weights=wb, minlength=nw)
        wk = wb * weights
        den += np.bincount(idx, weights=wk, minlength=nw)
        for ch in range(C):
            num[:, ch] += np.bincount(idx, weights=wk * values[:, ch], minlength=nw)
    return num, den, mass


def surface_lowpass(surface, values, weights, iterations=3):
    """Low-pass a per-texel signal on the welded mesh graph (seam-continuous, never across
    unrelated surfaces): weighted splat to vertices, normalised graph smoothing, interpolate."""
    num, den, _ = _splat(surface, values, weights)
    W = surface.topology.adjacency
    deg = np.maximum(np.asarray(W.sum(1)).ravel(), 1e-12)
    for _ in range(iterations):
        num = 0.5 * num + 0.5 * (W @ num) / deg[:, None]
        den = 0.5 * den + 0.5 * (W @ den) / deg
    vert = num / np.maximum(den, 1e-12)[:, None]
    vert[den < 1e-9] = 0.0
    return _interp_vertices(surface, vert)


# --------------------------------------------------------------------------------------
# completion
# --------------------------------------------------------------------------------------

def _vertex_field(surface, colour, alpha):
    """Harmonic surface fill on welded vertices. Returns (Vw,C) values + stats."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve, cg
    from scipy.sparse.csgraph import connected_components
    topo = surface.topology
    nw = topo.n_weld
    C = colour.shape[1]
    A, B, M = _splat(surface, colour, alpha)
    known = (M > 0) & (B >= 0.5 * M) & (B > 1e-6)
    x = np.zeros((nw, C))
    x[known] = A[known] / B[known, None]
    stats = {"weld_vertices": int(nw), "known_vertices": int(known.sum())}
    if not known.any():
        observed = alpha > 0
        fallback = (colour[observed].mean(0) if observed.any() else np.full(C, 0.5))
        x[:] = fallback
        stats["fallback"] = "no observed vertex; constant fill"
        return x, stats
    W = topo.adjacency
    unknown = ~known
    if unknown.any():
        U = np.flatnonzero(unknown)
        K = np.flatnonzero(known)
        W_U = W[U]
        W_UU = W_U[:, U]
        W_UK = W_U[:, K]
        anchored = np.asarray(W_UK.sum(1)).ravel() > 0
        ncomp, labels = connected_components(W_UU, directed=False)
        comp_anchor = np.bincount(labels, weights=anchored.astype(np.float64), minlength=ncomp) > 0
        floating = ~comp_anchor[labels]
        if floating.any():
            # Surface pieces never connected to an observation (separate inner parts):
            # nearest observed vertex in 3D.
            from scipy.spatial import cKDTree
            tree = cKDTree(topo.weld_pos[K])
            _, nn = tree.query(topo.weld_pos[U[floating]])
            x[U[floating]] = x[K[nn]]
            stats["floating_vertices"] = int(floating.sum())
            fixed = known.copy()
            fixed[U[floating]] = True
            U = np.flatnonzero(~fixed)
            K = np.flatnonzero(fixed)
            if U.size:
                W_U = W[U]
                W_UU = W_U[:, U]
                W_UK = W_U[:, K]
        if U.size:
            deg = np.asarray(W_U.sum(1)).ravel()
            L = (diags(deg) - W_UU).tocsc()
            rhs = W_UK @ x[K]
            if U.size <= 250_000:
                sol = spsolve(L, rhs)
                sol = np.asarray(sol).reshape(U.size, C)
                stats["solver"] = "superlu"
            else:
                sol = np.zeros((U.size, C))
                pre = diags(1.0 / np.maximum(deg, 1e-12))
                for ch in range(C):
                    sol[:, ch], _ = cg(L, rhs[:, ch], rtol=1e-6, maxiter=1500, M=pre)
                stats["solver"] = "cg"
            x[U] = sol
        stats["solved_vertices"] = int(U.size)
    return x, stats


def _free_neighbours(surface, free_idx):
    """4-neighbour texel indices (-1 = none) of the free texels, restricted to the same chart."""
    r = surface.raster
    T = r.size
    grid = np.full(T * T, -1, np.int32)
    grid[r.texel] = np.arange(r.texel.size, dtype=np.int32)
    tx = r.texel[free_idx] % T
    ty = r.texel[free_idx] // T
    chart = surface.texel_chart
    own = chart[free_idx]
    nb = np.full((free_idx.size, 4), -1, np.int64)
    for k, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        xx = tx + dx
        yy = ty + dy
        inb = (xx >= 0) & (xx < T) & (yy >= 0) & (yy < T)
        cand = np.full(free_idx.size, -1, np.int64)
        cand[inb] = grid[yy[inb] * T + xx[inb]]
        okc = cand >= 0
        okc[okc] = chart[cand[okc]] == own[okc]
        nb[okc, k] = cand[okc]
    return nb, ((tx + ty) & 1).astype(bool)


def complete_texels(surface, colour, alpha, device, band=BLEND_BAND, iterations=None):
    """Fill unobserved texels and blend weak observations. Returns (values, stats).

    1. harmonic membrane fill on welded mesh vertices (seam-continuous, low frequency);
    2. texels within `band` texels of a trusted observation are relaxed (red/black SOR,
       chart-restricted) so the fill meets the observed colours without a visible border;
    3. weak observations are blended with the fill by their confidence."""
    torch = _torch()
    t0 = time.perf_counter()
    colour = np.asarray(colour, np.float32)
    alpha = np.asarray(alpha, np.float32)
    x, stats = _vertex_field(surface, colour, alpha)
    t1 = time.perf_counter()
    strong = alpha >= 1.0 - 1e-6
    free_idx = np.flatnonzero(~strong)
    out = colour.copy()
    if free_idx.size == 0:
        stats.update(vertex_field_seconds=round(t1 - t0, 4), completed_texels=0)
        return out, stats
    vals = colour.copy()
    vals[free_idx] = _interp_vertices(surface, x, free_idx)
    nb, parity = _free_neighbours(surface, free_idx)
    # distance (in texel steps, inside the chart) from the trusted observations
    n = colour.shape[0]
    local = np.full(n, -1, np.int64)
    local[free_idx] = np.arange(free_idx.size)
    nb_local = np.where(nb >= 0, local[np.maximum(nb, 0)], -1)
    dist = np.full(free_idx.size, band + 1, np.int16)
    seeds = ((nb >= 0) & strong[np.maximum(nb, 0)]).any(1)
    dist[seeds] = 1
    frontier = np.flatnonzero(seeds)
    for k in range(2, band + 1):
        if frontier.size == 0:
            break
        cand = nb_local[frontier].ravel()
        cand = cand[cand >= 0]
        cand = np.unique(cand[dist[cand] > k])
        dist[cand] = k
        frontier = cand
    inband = dist <= band
    if iterations is None:
        iterations = 2 * band
    if inband.any() and iterations > 0:
        omega = 1.7
        vt = torch.from_numpy(vals).to(device)
        groups = []
        for p in (False, True):
            sel = inband & (parity == p) & (nb >= 0).any(1)
            if sel.any():
                ids = torch.from_numpy(free_idx[sel]).to(device)
                nbs = torch.from_numpy(nb[sel]).to(device)
                valid = (nbs >= 0).to(torch.float32)[..., None]
                cnt = valid.sum(1).clamp_min(1)
                groups.append((ids, nbs.clamp_min(0), valid, cnt))
        for it in range(iterations):
            if it % 16 == 0:
                _interrupt()
            for ids, nbs, valid, cnt in groups:
                mean = (vt[nbs] * valid).sum(1) / cnt
                vt[ids] = (1 - omega) * vt[ids] + omega * mean
        vals = vt.cpu().numpy()
    t2 = time.perf_counter()
    if strong.any():
        lo = np.minimum(colour[strong].min(0), 0.0)
        hi = np.maximum(colour[strong].max(0), 1.0)
    else:
        lo, hi = 0.0, 1.0
    h = np.clip(vals[free_idx], lo, hi)
    a = alpha[free_idx][:, None]
    out[free_idx] = a * colour[free_idx] + (1 - a) * h
    stats.update(vertex_field_seconds=round(t1 - t0, 4), texel_relax_seconds=round(t2 - t1, 4),
                 relax_iterations=int(iterations), blend_band_texels=int(inband.sum()),
                 completed_texels=int(free_idx.size))
    return out, stats


# --------------------------------------------------------------------------------------
# atlas assembly
# --------------------------------------------------------------------------------------

def to_atlas(surface, values, fill_gutter=True, background=0.0):
    """Scatter per-texel values to a (T,T,C) atlas; gutters take the nearest occupied texel."""
    r = surface.raster
    T = r.size
    values = np.asarray(values, np.float32)
    C = 1 if values.ndim == 1 else values.shape[1]
    img = np.full((T * T, C), background, np.float32)
    img[r.texel] = values.reshape(-1, C)
    if fill_gutter:
        img = img[surface.gutter_source()]
    return img.reshape(T, T, C)


def coverage_atlas(surface, alpha):
    r = surface.raster
    T = r.size
    out = np.zeros(T * T, np.float32)
    out[r.texel] = alpha
    return out.reshape(T, T)


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------

def choose_device(preference="auto", estimate_bytes=0):
    torch = _torch()
    if preference == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu"), "cpu requested" if preference == "cpu" else "no CUDA device"
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
        if getattr(device, "type", "cpu") != "cuda":
            return torch.device("cpu"), "ComfyUI device is not CUDA"
        try:
            mm.free_memory(int(estimate_bytes * 1.3) + (256 << 20), device)
        except Exception:
            pass
    except Exception:
        device = torch.device("cuda")
    try:
        free, _ = torch.cuda.mem_get_info(device)
        if free < estimate_bytes * 1.2:
            return torch.device("cpu"), f"only {free / 2**20:.0f} MiB VRAM free"
    except Exception:
        pass
    return device, "cuda"


VIEW_CACHE_BYTES = {"cuda": 512 << 20, "cpu": 1536 << 20}


def _estimate_bytes(n_texels, max_depth_px, max_image_px, chunk=1 << 20, detail=False):
    """Peak device memory: ~96 B per sample (geometry + accumulators), one or more prepared
    views (bounded by the view cache) and the per-chunk temporaries. The detail mode adds the
    fine-band accumulators and the best camera's sample (28 B per sample) and, for the one view
    being fused, the fine-band image (12 B/px), the 2x2 min/max images (24 B/px), the cubic-window
    map (1 B/px) and the temporaries while these are built (36 B/px). Measured with the PyTorch
    profiler (4096 bake from twelve 2048 views, views not cached): peak 1.29 GiB standard and
    1.78 GiB detail, against estimates of 1.73 and 2.42 GiB."""
    per_texel = 124 if detail else 96
    per_view = max_depth_px ** 2 * 4 + int(max_image_px ** 2 * 4 * 4 * 1.34) + max_image_px ** 2 * 4
    views = max(per_view, VIEW_CACHE_BYTES["cuda"])
    if detail:
        views += max_image_px ** 2 * (3 * 4 + 2 * 3 * 4 + 1 + 36)
    return int(n_texels * per_texel + views + chunk * 4 * (90 if detail else 60))


def bake_branch(surface, collection, semantic, device, completion=True, harmonize=True, timer=None,
                detail=False):
    """Bake one semantic ('base_color' or 'mr') from a view collection on a shared surface.
    detail=True (Base Color only) uses cubic reconstruction + a near single-camera fine band;
    visibility, confidence, coverage and completion are identical to the standard path."""
    torch = _torch()
    timer = timer or StageTimer()
    detail = bool(detail and semantic == "base_color")
    views = collect_views(collection)
    if semantic == "base_color" and harmonize and len(views) > 1:
        # Both passes need every view: keep the prepared views when they fit the budget.
        need = sum(_view_bytes(v, semantic) for v in views)
        budget = VIEW_CACHE_BYTES["cuda" if device.type == "cuda" else "cpu"]
        if need <= budget:
            views = [prepare_view(v, semantic, device) for v in views]
            timer.lap(f"{semantic}:prepare_views")
    geo = texel_geometry(surface, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timer.lap(f"{semantic}:texel_geometry")
    gains = None
    gain_report = {}
    if semantic == "base_color" and harmonize and len(views) > 1:
        gains, gain_report = estimate_gains(geo, views, device)
        timer.lap(f"{semantic}:view_harmonisation")
    fused = fuse_views(surface, geo, views, semantic, device, gains, detail=detail)
    sharp, broad, qmax, nviews, per_view, p_sharp = fused[:6]
    fine, best_sharp = fused[6] if detail else (None, None)
    del fused
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timer.lap(f"{semantic}:projection_visibility_fusion")
    sharp = np.nan_to_num(sharp.cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0)
    broad = np.nan_to_num(broad.cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0)
    qmax = qmax.cpu().numpy()
    nviews = nviews.cpu().numpy()
    del geo
    alpha = _smoothstep(qmax, Q_LO, Q_HI)
    seen = qmax >= Q_LO
    # two-band blend: detail from the sharp fusion, colour transitions from the broad one
    diff = np.where(seen[:, None], broad - sharp, 0.0)
    colour = np.where(seen[:, None], sharp + surface_lowpass(surface, diff, seen.astype(np.float64)), 0.0)
    del diff, broad
    fusion_info = {"sharp_power": p_sharp, "broad_power": FUSION_POWER_BROAD}
    if fine is not None:
        # detail mode: add the fine band (cubic reconstruction minus G(1 px)) of ~the best camera.
        # Never extrapolate beyond both the fused colour and the best camera's own sharp sample
        # (misregistered views would otherwise create darker/brighter rims than any view shows).
        fine = np.nan_to_num(fine.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        best_sharp = np.nan_to_num(best_sharp.cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0)
        lo = np.minimum(colour, best_sharp) - DETAIL_RANGE_SLACK
        hi = np.maximum(colour, best_sharp) + DETAIL_RANGE_SLACK
        colour = np.where(seen[:, None], np.clip(colour + fine, lo, hi), 0.0)
        del fine, best_sharp, lo, hi
        fusion_info.update(mode="detail", fine_band_sigma_view_px=DETAIL_SIGMA_FINE,
                           fine_band_power=DETAIL_POWER_FINE, reconstruction="cubic (Keys -0.75) + anti-ringing")
    colour = np.clip(colour, 0.0, None).astype(np.float32)
    timer.lap(f"{semantic}:two_band_blend")
    area = surface.texel_area.astype(np.float64)
    total_area = max(float(area.sum()), 1e-30)
    stats = {
        "texels": int(alpha.size),
        "observed_strong_fraction": float((alpha >= 1 - 1e-6).mean()),
        "observed_any_fraction": float((qmax >= Q_LO).mean()),
        "unobserved_fraction": float((qmax < Q_LO).mean()),
        "observed_area_fraction": float((area * alpha).sum() / total_area),
        "mean_views_per_observed_texel": float(nviews[qmax >= Q_LO].mean()) if (qmax >= Q_LO).any() else 0.0,
        "fusion": fusion_info,
        "views": per_view,
    }
    if gains is not None:
        stats["view_gains_rgb"] = [[round(float(c), 4) for c in g] for g in gains]
        stats["harmonisation"] = gain_report
    if completion:
        values, cstats = complete_texels(surface, colour, alpha, device)
        stats["completion"] = cstats
    else:
        # Diagnostic mode: unseen Base Color stays black; unseen MR gets a neutral rough dielectric
        # (never roughness 0, which would look like a mirror).
        empty = np.zeros(colour.shape[1], np.float32) if semantic == "base_color" else np.array([0.8, 0.0], np.float32)
        values = np.where((qmax >= Q_LO)[:, None], colour, empty).astype(np.float32)
    timer.lap(f"{semantic}:completion")
    if semantic == "base_color":
        values = linear_to_srgb_np(values)
    values = np.clip(values, 0.0, 1.0)
    return values, alpha, stats


def bake_surface_maps(albedo_setup, albedo_views, size, completion=True, mr_setup=None, mr_views=None,
                      device_preference="auto", harmonize=True, detail=False):
    """Bake Base Color (+ Roughness/Metallic) atlases. Returns (maps, coverage, report).
    detail=True applies the detail-preserving reconstruction/fusion to Base Color only;
    Roughness/Metallic always use the standard path (scalar material data)."""
    torch = _torch()
    timer = StageTimer()
    context = albedo_setup.context
    if mr_views is not None and mr_setup is not None and mr_setup.context.atlas_id != context.atlas_id:
        raise ValueError("MR setup uses a different UV layout than Albedo")
    surface, cache_hit = prepare_surface(context, int(size))
    timer.lap("uv_raster_and_topology")
    n = surface.raster.texel.size + surface.raster.extra_face.size
    depth_px = max(int(vs.gbuffer.depths[0].shape[0]) for vs in albedo_views.sets)
    image_px = max(int(vs.images[0].shape[0]) for vs in albedo_views.sets)
    if mr_views is not None:
        depth_px = max(depth_px, *(int(vs.gbuffer.depths[0].shape[0]) for vs in mr_views.sets))
        image_px = max(image_px, *(int(vs.images[0].shape[0]) for vs in mr_views.sets))
    device, device_note = choose_device(device_preference, _estimate_bytes(n, depth_px, image_px, detail=detail))
    policy=albedo_setup.runtime
    if device.type=='cuda' and getattr(policy,'performance_mode','original')!='original':
        from .adaptive import snapshot,decide,MiB
        state=snapshot(device)
        decision=decide(state,policy)
        estimate=_estimate_bytes(n,depth_px,image_px,detail=detail)
        if state is not None:
            if estimate+decision.headroom>decision.capacity:
                device,device_note=_torch().device('cpu'),'adaptive bake budget: native CPU fallback'
            else:
                import comfy.model_management as mm
                mm.free_memory(int(estimate*1.3)+decision.headroom+max(0,state.total-decision.capacity),device)
                refreshed=decide(snapshot(device),policy)
                if estimate*1.2>refreshed.spendable:
                    device,device_note=_torch().device('cpu'),'adaptive bake pressure: native CPU fallback'
    timer.lap("device_select")
    report = {"stage": "fast_surface_bake", "texture_size": int(size), "device": str(device),
              "device_note": device_note, "surface_cache_hit": cache_hit,
              "occupied_texels": int(surface.raster.texel.size), "uv_charts": int(surface.topology.n_charts),
              "conservative_texels": int(surface.raster.conservative),
              "overlapping_uv_samples": int(surface.raster.extra_face.size),
              "detail_mode": bool(detail)}
    if not cache_hit:
        report["surface_prepare_seconds"] = surface.timings

    def run(dev):
        maps = {}
        with torch.inference_mode():
            base, alpha, stats = bake_branch(surface, albedo_views, "base_color", dev, completion, harmonize, timer,
                                             detail=detail)
            report["base_color"] = stats
            maps["base_color"] = to_atlas(surface, base, fill_gutter=True)
            coverage = coverage_atlas(surface, alpha)
            timer.lap("base_color:atlas_and_gutter")
            if mr_views is not None:
                mr, _, mr_stats = bake_branch(surface, mr_views, "mr", dev, completion, False, timer)
                report["mr"] = mr_stats
                atlas = to_atlas(surface, mr, fill_gutter=True)
                maps["roughness"] = np.repeat(atlas[..., 0:1], 3, axis=2)
                maps["metallic"] = np.repeat(atlas[..., 1:2], 3, axis=2)
                timer.lap("mr:atlas_and_gutter")
        return maps, coverage

    try:
        maps, coverage = run(device)
    except Exception as error:  # CUDA out of memory: finish on the CPU instead of failing the graph
        if device.type != "cuda" or "out of memory" not in str(error).lower():
            raise
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        device = torch.device("cpu")
        report["device"] = "cpu"
        report["device_note"] = "CUDA ran out of memory; finished on CPU"
        maps, coverage = run(device)
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    report["timings_seconds"] = dict(timer.stages)
    report["bake_total_seconds"] = timer.total()
    return maps, coverage, report
