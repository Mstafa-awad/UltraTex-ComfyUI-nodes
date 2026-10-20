"""Preserve calibrated front-view RGB; other views cover its unreliable regions.

No semantic paint mask, image warp, or learned dependency is involved. View 0
is copied only through the geometry-tested sampler. Its strong observations
are not recoloured or averaged with the other generated images.
"""
import numpy as np
from . import fastbake as fb

def anchor_weight(q, fallback_q, torch):
    t = ((q - 0.12) / 0.18).clamp(0, 1)
    a = t * t * (3 - 2 * t)
    return torch.where(fallback_q < fb.Q_LO, (q >= fb.Q_LO).float(), a)

def bake(surface, collection, device, completion=True, timer=None, exact_support=False):
    from .sharpbake import attach_source_base, proxy_view
    from .enhance import restore_original
    torch = fb._torch()
    timer = timer or fb.StageTimer()
    views = fb.collect_views(collection)
    geo = fb.texel_geometry(surface, device, analytic_footprint=True)
    n = len(surface.raster.texel)
    full = torch.zeros((n, 3), device=device)
    weight = torch.zeros(n, device=device)
    best = torch.zeros(n, device=device)
    counts = torch.zeros(n, dtype=torch.int16, device=device)
    timer.lap('base_color:anchor_geometry')
    originals = fb.collect_views(restore_original(collection))
    originals = [proxy_view(v) if v.image_ref.shape[0] > 2048 else v for v in originals]
    gains, gain_report = fb.estimate_gains(geo, originals, device, complete_mips=True)
    # Keep the anchor exposure exact; harmonize only fallback cameras.
    gains = np.clip(gains / np.maximum(gains[0], 1e-08), 1 / 1.15, 1.15)
    gains[0] = 1
    timer.lap('base_color:anchor_colour_matching')
    per = []
    chunk = 1 << 18

    def add(c, q, positions):
        if isinstance(positions, slice):
            old = best[positions]
            new = torch.maximum(old, q)
            scale = (old / new.clamp_min(1e-20)).pow(64)
            w = (q / new.clamp_min(1e-20)).pow(64)
            full[positions] = full[positions] * scale[:, None] + c * w[:, None]
            weight[positions] = weight[positions] * scale + w
            best[positions] = new
        else:
            unique = torch.unique(positions)
            old = best[unique].clone()
            best.scatter_reduce_(0, positions, q, reduce='amax', include_self=True)
            scale = (old / best[unique].clamp_min(1e-20)).pow(64)
            full[unique] *= scale[:, None]
            weight[unique] *= scale
            w = (q / best[positions].clamp_min(1e-20)).pow(64)
            full.index_add_(0, positions, c * w[:, None])
            weight.index_add_(0, positions, w)
    for vi in list(range(1, len(views))) + [0]:
        fb._interrupt()
        pv = attach_source_base(fb.prepare_view(views[vi], 'base_color', device, complete_mips=True, stream=True, surface_guard=True))
        pv.exact_filter_support = bool(exact_support)
        gain = torch.tensor(gains[vi], device=device)
        used = anchored = 0
        # Secondary UV claims receive no unrelated primary-camera preference.
        for s in range(n, len(geo.P), chunk):
            e = min(len(geo.P), s + chunk)
            c, q, _ = fb._project(geo.slice(s, e), pv, torch, source_detail=True)
            add(c * gain, q, geo.extra_pos[s - n:e - n])
        if vi == 0:
            full.div_(weight.clamp_min(1e-20)[:, None])
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            c, q, _ = fb._project(geo.slice(s, e), pv, torch, source_detail=True)
            if vi:
                add(c * gain, q, slice(s, e))
            else:
                a = anchor_weight(q, best[s:e], torch)
                full[s:e] = c * a[:, None] + full[s:e] * (1 - a[:, None])
                best[s:e] = torch.maximum(best[s:e], q)
                anchored += int((a >= 0.999999).sum())
            counts[s:e] += (q >= fb.Q_LO).to(torch.int16)
            used += int((q >= fb.Q_LO).sum())
        per.append(dict(view=pv.label, texels_used=used, fully_anchored_texels=anchored, guard_candidates=getattr(pv, 'guard_candidates', 0), guard_rejected=getattr(pv, 'guard_rejected', 0)))
        del pv
    colour = np.nan_to_num(full.cpu().numpy(), nan=0, posinf=1, neginf=0)
    q = best.cpu().numpy()
    counts = counts.cpu().numpy()
    del geo, full, weight, best
    timer.lap('base_color:anchor_projection')
    alpha = fb._smoothstep(q, fb.Q_LO, fb.Q_HI)
    area = surface.texel_area.astype(np.float64)
    stats = dict(texels=n, observed_strong_fraction=float((alpha >= 1 - 1e-06).mean()), observed_any_fraction=float((q >= fb.Q_LO).mean()), unobserved_fraction=float((q < fb.Q_LO).mean()), observed_area_fraction=float((area * alpha).sum() / max(area.sum(), 1e-30)), mean_views_per_observed_texel=float(counts[q >= fb.Q_LO].mean()) if (q >= fb.Q_LO).any() else 0, fusion=dict(mode='sharp_anchor_v2' if exact_support else 'sharp_anchor', anchor_view=views[0].label, anchor_full_quality=0.3, anchor_transition_start=0.12, anchor_colour_unchanged=True, source_images_warped=False, filter_support='exact mip/cubic texel bounds' if exact_support else 'rc3 radius estimate'), views=per, view_gains_rgb=gains.tolist(), harmonisation=gain_report)
    if completion:
        values, stats['completion'] = fb.complete_texels(surface, colour, alpha, device)
    else:
        values = np.where((q >= fb.Q_LO)[:, None], colour, 0)
    timer.lap('base_color:completion')
    return (np.clip(fb.linear_to_srgb_np(values), 0, 1), alpha, stats)
