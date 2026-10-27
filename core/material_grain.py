"""Example-based material detail on one shared 3D field, restricted by an atlas mask.

Independent NumPy/SciPy implementation. No learned weights or external synthesis
code. Only the masked material's small-scale appearance is replaced.
"""
import numpy as np
from scipy import ndimage as ndi

from . import fastbake as fb
from .runtime import interrupt


def _cut(cost):
    total = cost.copy()
    parent = np.zeros(cost.shape, np.int8)
    for y in range(1, len(cost)):
        previous = np.pad(total[y - 1], 1, constant_values=np.inf)
        choices = np.stack((previous[:-2], previous[1:-1], previous[2:]))
        parent[y] = choices.argmin(0) - 1
        total[y] += choices.min(0)
    x = int(total[-1].argmin())
    path = np.empty(len(cost), np.int32)
    for y in range(len(cost) - 1, -1, -1):
        path[y] = x
        x += int(parent[y, x])
    return path


def quilt(example, size=768, seed=0):
    """Copy overlapping example patches, joining them with minimum-error cuts."""
    rng = np.random.default_rng(seed)
    patch = min(48, min(example.shape))
    overlap = patch // 3
    step = patch - overlap
    result = np.zeros((size + patch, size + patch), np.float32)
    for y in range(0, size, step):
        interrupt()
        for x in range(0, size, step):
            ys = rng.integers(0, example.shape[0] - patch + 1, 48)
            xs = rng.integers(0, example.shape[1] - patch + 1, 48)
            candidates = np.stack([example[a:a+patch, b:b+patch] for a, b in zip(ys, xs)])
            for i in range(len(candidates)):
                candidates[i] = np.rot90(candidates[i], int(rng.integers(4)))
                if rng.integers(2):
                    candidates[i] = candidates[i, :, ::-1]
            old = result[y:y+patch, x:x+patch]
            error = np.zeros(len(candidates), np.float32)
            if x:
                error += ((candidates[:, :, :overlap] - old[None, :, :overlap]) ** 2).mean((1, 2))
            if y:
                error += ((candidates[:, :overlap] - old[None, :overlap]) ** 2).mean((1, 2))
            chosen = candidates[int(error.argmin())]
            take = np.ones((patch, patch), bool)
            if x:
                path = _cut((chosen[:, :overlap] - old[:, :overlap]) ** 2)
                take[:, :overlap] = np.arange(overlap)[None] >= path[:, None]
            if y:
                path = _cut(((chosen[:overlap] - old[:overlap]) ** 2).T)
                take[:overlap] &= np.arange(overlap)[:, None] >= path[None]
            old[take] = chosen[take]
    result = result[:size, :size].copy()
    result -= result.mean()
    return result


def grain_tiles(reference, seed=0):
    reference = np.asarray(reference, np.float32)
    if reference.ndim != 3 or reference.shape[2] < 3 or min(reference.shape[:2]) < 48:
        raise ValueError('Grain example must be an RGB material crop at least 48 by 48 pixels')
    if not np.isfinite(reference).all() or reference.min() < 0 or reference.max() > 1:
        raise ValueError('Grain example must contain finite RGB values in [0,1]')
    luminance = reference[..., :3] @ np.float32([.2126, .7152, .0722])
    log_light = np.log(np.maximum(luminance, .005))
    residual = log_light - ndi.gaussian_filter(log_light, 5.)
    return [quilt(residual, seed=seed+i) for i in range(3)], np.quantile(residual, [.005, .995])


def sample_field(positions, normals, tiles, density):
    """One surface-stable triplanar field, with detail variance retained in overlaps."""
    weights = np.abs(normals).astype(np.float32) ** 4
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
    detail = np.zeros(len(positions), np.float32)
    for axis, (a, b) in enumerate(((1, 2), (0, 2), (0, 1))):
        values = ndi.map_coordinates(tiles[axis],
            [positions[:, a]*density, positions[:, b]*density], order=1, mode='reflect', prefilter=False)
        detail += values * weights[:, axis]
    return detail / np.maximum(np.sqrt((weights**2).sum(1)), 1e-6)


def repair_atlas(context, atlas, material_mask, reference, density=4000., strength=1., seed=0,reference_color_strength=0.):
    """Retain unmasked RGB exactly; reconstruct masked grain in shared mesh coordinates."""
    atlas = np.asarray(atlas, np.float32)
    mask = np.asarray(material_mask, np.float32)
    if atlas.ndim != 3 or atlas.shape[2] != 3 or atlas.shape[0] != atlas.shape[1]:
        raise ValueError('Material grain repair requires one square RGB atlas')
    if mask.shape != atlas.shape[:2] or not np.isfinite(mask).all():
        raise ValueError('Material mask must be finite and match the atlas resolution and UV layout')
    if not np.isfinite(atlas).all() or atlas.min() < 0 or atlas.max() > 1:
        raise ValueError('Atlas must contain finite RGB values in [0,1]')
    if not 0 <= strength <= 1 or not 100 <= density <= 20000 or not 0 <= reference_color_strength <= 1:
        raise ValueError('Invalid material grain strength or density')
    result = atlas.copy()
    if strength == 0 or not np.any(mask > 0):
        return result, {'modified_texels': 0}
    tiles, limits = grain_tiles(reference, seed)
    size = len(atlas)
    surface, _ = fb.prepare_surface(context, size)
    raster = surface.raster
    alpha = np.clip(mask.reshape(-1)[raster.texel], 0, 1) * strength
    active = np.flatnonzero(alpha > 0)
    faces = surface.topology.faces
    values = result.reshape(-1, 3)
    # Sharing only grain leaves the per-chart color patches intact. Estimate
    # the masked material's base on the welded surface too.
    shared_base=fb.surface_lowpass(surface,atlas.reshape(-1,3)[raster.texel],alpha,iterations=2)
    reference_linear=fb.srgb_to_linear_np(np.asarray(reference,np.float32)[...,:3])
    reference_chroma=np.median(reference_linear.reshape(-1,3),axis=0)
    luminance_weights=np.float32([.2126,.7152,.0722])
    reference_chroma/=max(float(reference_chroma@luminance_weights),1e-8)
    for chart in np.unique(surface.texel_chart[active]):
        interrupt()
        selected = active[surface.texel_chart[active] == chart]
        texels = raster.texel[selected]
        base = shared_base[selected]
        for start in range(0, len(selected), 1 << 18):
            end = min(len(selected), start+(1 << 18))
            ids = selected[start:end]
            vertex_ids = faces[raster.face[ids]]
            bary = raster.bary[ids]
            position = (context.vertices[vertex_ids] * bary[..., None]).sum(1)
            normal = (context.normals[vertex_ids] * bary[..., None]).sum(1)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            detail = np.clip(sample_field(position, normal, tiles, density), *limits)
            repaired = base[start:end] * np.exp(detail)[:, None]
            if reference_color_strength:
                linear=fb.srgb_to_linear_np(np.clip(repaired,0,1))
                light=linear@luminance_weights
                matched=light[:,None]*reference_chroma[None]
                linear=linear*(1-reference_color_strength)+matched*reference_color_strength
                repaired=fb.linear_to_srgb_np(np.clip(linear,0,1))
            a = alpha[ids, None]
            target = raster.texel[ids]
            values[target] = np.clip(atlas.reshape(-1, 3)[target]*(1-a) + repaired*a, 0, 1)
    occupied = np.zeros((size, size), bool)
    occupied.reshape(-1)[raster.texel] = True
    nearest = ndi.distance_transform_edt(~occupied, return_distances=False, return_indices=True)
    gutter = (~occupied) & (mask[nearest[0], nearest[1]] > 0)
    result[gutter] = result[nearest[0][gutter], nearest[1][gutter]]
    return result, {'modified_texels': int(len(active)), 'grain_density': density,
        'strength': strength, 'seed': seed, 'detail_source': 'provided material example',
        'coordinates': 'shared 3D field; independent of camera and UV chart',
        'material_base': 'masked color field on the welded surface; continuous across UV charts',
        'reference_color_strength': reference_color_strength,
        'unmasked_surface_texels_unchanged': True, 'gutter_texels_updated': int(gutter.sum())}
