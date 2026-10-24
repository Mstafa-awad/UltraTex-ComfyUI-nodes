"""Geometry-tested multi-camera confirmation of suspicious source RGB.

No asset palette, semantic region, UV location or character identity is used.
An observation supported by two independent calibrated cameras is retained. Correction
requires two independent cameras to agree on a different surface colour.

The computation is vectorised over the candidate pixels with PyTorch and runs on the device of
``candidates`` (CUDA in a bake, CPU in the tests). Each alternative camera is uploaded, sampled for every
candidate and released before the next one, so only one set of full-size G-buffers is resident at a time.
"""
import warnings

import numpy as np

FOCAL = 1.5625
CHUNK = 1 << 18          # candidate pixels per vectorised block (bounds the temporaries)
OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _rgb(ref):
    a = np.asarray(ref.read())
    return a, 255. if a.dtype == np.uint8 else 1.


def _upload(array, torch, device):
    """Copy a (memory-mapped) array to the device once, without an intermediate host copy."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # read-only memory map: it is only read here
        return torch.from_numpy(np.asarray(array)).to(device)


def _calibrate(colour, gain, torch):
    if gain is None:
        return colour
    linear = torch.where(colour <= .04045, colour / 12.92, ((colour + .055) / 1.055) ** 2.4) * torch.as_tensor(
        np.asarray(gain, np.float32), device=colour.device)
    return torch.where(linear <= .0031308, 12.92 * linear, 1.055 * linear.clamp_min(0) ** (1 / 2.4) - .055)


def _normals(normal, ty, tx, rotation):
    n = (normal[ty, tx, :3].float() / 127.5 - 1) @ rotation.T
    return n


def confirm(view, others, candidates, torch, gains=None, enhanced_chroma=False, return_reference=False):
    """Return confirmed outliers in the ORIGINAL source pixel grid.

    `candidates` is a boolean (size,size) tensor; the result is a tensor of the same shape and device.
    With `return_reference` a (size,size,3) float16 array of the agreeing cameras' colour is returned too.
    """
    device = candidates.device
    result = torch.zeros_like(candidates)

    def empty(stats):
        if return_reference:
            return result, stats, np.zeros((*candidates.shape, 3), np.float16)
        return result, stats
    if view.source_ref is None or view.source_mask_ref is None:
        return empty({'tested': 0, 'confirmed': 0})
    src, scale = _rgb(view.source_ref)
    size = len(src)
    if candidates.shape != (size, size):
        raise ValueError('Photo-consistency must use original source pixels')
    ids = candidates.reshape(-1).nonzero(as_tuple=True)[0]
    count = int(ids.numel())
    if not count:
        return empty({'tested': 0, 'confirmed': 0})
    if view.normal_ref is None or view.normal_rotation is None:
        return empty({'tested': count, 'confirmed': 0, 'reason': 'surface normals unavailable'})
    records = []
    for other in others:
        if other is view or np.allclose(other.matrix, view.matrix, atol=1e-7):
            continue
        if other.source_ref is None or other.source_mask_ref is None:
            continue
        expected = other.transform.original_width if other.transform is not None else other.image_ref.shape[0]
        if _shape(other.source_ref)[:2] != (expected, expected):
            continue
        records.append(other)
    if len(records) < 2:
        return empty({'tested': count, 'confirmed': 0, 'reason': 'fewer than two independent alternative cameras'})

    f32 = torch.float32
    depth = _upload(view.depth_ref.read(), torch, device)
    R = len(depth)
    own_normal = _upload(view.normal_ref.read(), torch, device)
    own_rotation = torch.as_tensor(np.asarray(view.normal_rotation, np.float32), device=device)
    own_foreground = _upload(view.source_mask_ref.read(), torch, device) > .5
    own_src = _upload(src, torch, device)
    inverse = np.linalg.inv(np.asarray(view.matrix, np.float64))
    inverse_rotation = torch.as_tensor(inverse[:3, :3].T.copy(), dtype=f32, device=device)
    inverse_translation = torch.as_tensor(inverse[:3, 3].copy(), dtype=f32, device=device)
    enhanced = enhanced_mask = None
    if enhanced_chroma:
        enhanced = np.asarray(view.image_ref.read())
        enhanced_mask = _upload(view.mask_ref.read(), torch, device)
    enhanced_scale = 255. if enhanced is not None and enhanced.dtype == np.uint8 else 1.

    # ---- the suspect camera's own pixels -------------------------------------------------------------------
    current = torch.empty((count, 3), dtype=f32, device=device)
    points_p = torch.empty((count, 3), dtype=f32, device=device)
    points_n = torch.empty((count, 3), dtype=f32, device=device)
    points_d = torch.empty(count, dtype=f32, device=device)
    contrast_detail = torch.zeros(count, dtype=torch.bool, device=device)
    for start in range(0, count, CHUNK):
        sel = ids[start:start + CHUNK]
        yy, xx = sel // size, sel % size
        u = (xx.to(f32) + .5) / size
        v = (yy.to(f32) + .5) / size
        colour = own_src[yy, xx, :3].to(f32) / scale
        if enhanced_chroma:
            # Trace the strongest SR chroma departure within this original pixel, using its actual
            # subpixel ray/depth, not the cell centre.
            base = colour - colour.mean(1, keepdim=True)
            best_drift = torch.full((len(sel),), -1., dtype=f32, device=device)
            colour = torch.full_like(colour, float('nan'))
            chosen_u, chosen_v = u.clone(), v.clone()
            E = len(enhanced)
            for oy in (-.375, 0, .375):
                for ox in (-.375, 0, .375):
                    su = (xx.to(f32) + .5 + ox) / size
                    sv = (yy.to(f32) + .5 + oy) / size
                    eu, ev = view.transform.uv(su, sv) if view.transform is not None else (su, sv)
                    ex = torch.floor(eu * E).long().clamp(0, E - 1)
                    ey = torch.floor(ev * E).long().clamp(0, E - 1)
                    sample = torch.from_numpy(enhanced[ey.cpu().numpy(), ex.cpu().numpy(), :3].astype(np.float32)).to(device) / enhanced_scale
                    delta = (sample - sample.mean(1, keepdim=True) - base).norm(dim=1)
                    take = (eu >= 0) & (eu < 1) & (ev >= 0) & (ev < 1) & (enhanced_mask[ey, ex] > .5) & (delta > best_drift)
                    colour[take] = sample[take]
                    best_drift[take] = delta[take]
                    chosen_u[take] = su[take]
                    chosen_v[take] = sv[take]
            u, v = chosen_u, chosen_v
        y = (v * R).long().clamp(max=R - 1)
        x = (u * R).long().clamp(max=R - 1)
        d = depth[y, x]
        pc = torch.stack(((u - .5) / FOCAL * d, (.5 - v) / FOCAL * d, -d), 1)
        dark = torch.zeros(len(sel), dtype=torch.bool, device=device)
        light = torch.zeros_like(dark)
        # Protect high-contrast achromatic designs (text, highlights, pupils, line art) without naming an
        # asset region or matching a colour palette.
        for dy in (-3, 0, 3):
            for dx in (-3, 0, 3):
                sy = (yy + dy).clamp(0, size - 1)
                sx = (xx + dx).clamp(0, size - 1)
                t = own_src[sy, sx, :3].to(f32) / scale
                inside = own_foreground[sy, sx]
                dark |= (t.amax(1) < .18) & inside
                light |= (t.amin(1) > .8) & ((t.amax(1) - t.amin(1)) < .10) & inside
        contrast_detail[start:start + CHUNK] = dark & light
        current[start:start + CHUNK] = _calibrate(colour, gains.get(view.label), torch) if gains is not None else colour
        points_p[start:start + CHUNK] = pc @ inverse_rotation + inverse_translation
        n = _normals(own_normal, y, x, own_rotation)
        points_n[start:start + CHUNK] = n / n.norm(dim=1, keepdim=True).clamp_min(1e-8)
        points_d[start:start + CHUNK] = d
    del own_src, own_normal, depth

    # ---- every alternative camera sees the same points -----------------------------------------------------
    cameras = len(records)
    observed = torch.full((cameras, count, 3), float('nan'), dtype=f32, device=device)
    found = torch.zeros((cameras, count), dtype=torch.bool, device=device)
    direction = torch.empty((cameras, count, 3), dtype=f32, device=device)
    for k, other in enumerate(records):
        im = _upload(other.source_ref.read(), torch, device)
        den = 255. if im.dtype == torch.uint8 else 1.
        td = _upload(other.depth_ref.read(), torch, device)
        fg = _upload(other.source_mask_ref.read(), torch, device) > .5
        has_normal = other.normal_ref is not None and other.normal_rotation is not None
        normal = _upload(other.normal_ref.read(), torch, device) if has_normal else None
        rotation = torch.as_tensor(np.asarray(other.normal_rotation, np.float32), device=device) if has_normal else None
        M = np.asarray(other.matrix, np.float64)
        Mr = torch.as_tensor(M[:3, :3].T.copy(), dtype=f32, device=device)
        Mt = torch.as_tensor(M[:3, 3].copy(), dtype=f32, device=device)
        camera = torch.as_tensor(np.linalg.inv(M)[:3, 3].copy(), dtype=f32, device=device)
        T, S = len(td), len(im)
        for start in range(0, count, CHUNK):
            part = slice(start, start + CHUNK)
            P, N, d = points_p[part], points_n[part], points_d[part]
            q = P @ Mr + Mt
            z = -q[:, 2]
            tu = .5 + FOCAL * q[:, 0] / z.clamp_min(1e-8)
            tv = .5 - FOCAL * q[:, 1] / z.clamp_min(1e-8)
            toward = camera - P
            toward = toward / toward.norm(dim=1, keepdim=True).clamp_min(1e-8)
            facing = (N * toward).sum(1)
            in_frame = (d > 0) & (z > .05) & (tu > 0) & (tu < 1) & (tv > 0) & (tv < 1) & (facing > .15)
            best = torch.full((len(d),), float('inf'), dtype=f32, device=device)
            chosen = torch.full((len(d), 3), float('nan'), dtype=f32, device=device)
            nc = N @ Mr
            num = (nc * q).sum(1)
            # Allow one original source pixel of alignment uncertainty. Every colour tap must still belong
            # to the same depth/normal surface.
            for oy, ox in OFFSETS:
                su = tu + ox / S
                sv = tv + oy / S
                tx = torch.floor(su * T).long().clamp(0, T - 1)
                ty = torch.floor(sv * T).long().clamp(0, T - 1)
                D = td[ty, tx]
                # Local tangent-plane prediction avoids treating a sloping surface's ordinary depth gradient
                # as an occluder.
                divisor = nc[:, 0] * ((tx.to(f32) + .5) / T - .5) / FOCAL + nc[:, 1] * (.5 - (ty.to(f32) + .5) / T) / FOCAL - nc[:, 2]
                steep = divisor.abs() > .03
                predicted = torch.where(steep, num / torch.where(steep, divisor, torch.ones_like(divisor)), z)
                same = in_frame & (D > 0) & ((D - predicted).abs() < .0002 + 2 * z / (FOCAL * T))
                if normal is not None:
                    nn = _normals(normal, ty, tx, rotation)
                    same &= (N * nn).sum(1) > .8 * nn.norm(dim=1).clamp_min(1e-8)
                ix = torch.floor(su * S).long().clamp(0, S - 1)
                iy = torch.floor(sv * S).long().clamp(0, S - 1)
                same &= (su >= 0) & (su < 1) & (sv >= 0) & (sv < 1) & fg[iy, ix]
                distance = ox * ox + oy * oy + (D - predicted).abs() * 100
                take = same & (distance < best)
                chosen[take] = im[iy[take], ix[take], :3].to(f32) / den
                best[take] = distance[take]
            if gains is not None:
                chosen = _calibrate(chosen, gains.get(other.label), torch)
            observed[k, part] = chosen
            found[k, part] = torch.isfinite(best)
            direction[k, part] = toward
        del im, td, fg, normal

    # ---- pairs of alternative cameras: do two independent cameras agree on a different colour? ---------------
    answer = torch.zeros(size * size, dtype=torch.bool, device=device)
    references = torch.zeros((size * size, 3), dtype=torch.float16, device=device) if return_reference else None
    supported_total = eligible_total = visible_total = 0
    tolerance = .08 if enhanced_chroma else .14
    for start in range(0, count, CHUNK):
        part = slice(start, start + CHUNK)
        own = current[part]
        different = torch.zeros(len(own), dtype=torch.bool, device=device)
        same_pair = torch.zeros_like(different)
        independent_pair = torch.zeros_like(different)
        best_reference = torch.zeros((len(own), 3), dtype=f32, device=device)
        best_cost = torch.full((len(own),), float('inf'), dtype=f32, device=device)
        if enhanced_chroma:
            own = own - own.mean(1, keepdim=True)
        for i in range(cameras):
            for j in range(i):
                raw_a, raw_b = observed[i, part], observed[j, part]
                a, b = raw_a, raw_b
                if enhanced_chroma:
                    a = a - a.mean(1, keepdim=True)
                    b = b - b.mean(1, keepdim=True)
                both = found[i, part] & found[j, part]
                independent = (direction[i, part] * direction[j, part]).sum(1) < .98
                independent_pair |= both & independent
                pair_distance = (a - b).norm(dim=1)
                from_a, from_b = (a - own).norm(dim=1), (b - own).norm(dim=1)
                source_distance = torch.minimum(from_a, from_b)
                # Generated cameras can differ slightly in exposure/shading. Permit that only when both are
                # much closer to each other than to the suspect observation, with an absolute cap.
                agree = (pair_distance < .12) | ((pair_distance < .22) & (pair_distance < .35 * source_distance))
                disagree = source_distance > .20
                if enhanced_chroma:
                    an, bn = a.norm(dim=1), b.norm(dim=1)
                    cosine = (a * b).sum(1) / (an * bn).clamp_min(1e-8)
                    agree = (pair_distance < .06) | ((cosine > .93) & (pair_distance < .20) & (an > .08) & (bn > .08))
                    disagree = source_distance > .12
                pair = both & independent & agree & disagree
                different |= pair
                same_pair |= both & independent & (from_a < tolerance) & (from_b < tolerance)
                take = pair & (pair_distance < best_cost)
                best_reference[take] = (raw_a[take] + raw_b[take]) * .5
                best_cost[take] = pair_distance[take]
        selected = different & ~same_pair & ~contrast_detail[part]
        answer[ids[part]] = selected
        if return_reference:
            references[ids[part][selected]] = best_reference[selected].half()
        supported_total += int(same_pair.sum())
        eligible_total += int(different.sum())
        visible_total += int(independent_pair.sum())
    result = answer.reshape(size, size)
    stats = dict(tested=count, visible_in_two_independent_cameras=visible_total,
                 alternative_agreement=eligible_total, source_supported_by_two_cameras=supported_total,
                 confirmed=int(answer.sum()))
    if return_reference:
        return result, stats, references.reshape(size, size, 3).cpu().numpy()
    return result, stats


def _shape(ref):
    return tuple(ref.read().shape)
