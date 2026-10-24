"""Preserve calibrated front-view RGB; other views cover its unreliable regions.

No semantic paint mask, image warp, or learned dependency is involved. View 0
is copied only through the geometry-tested sampler. Its strong observations
are not recoloured or averaged with the other generated images.

Fallback cameras are fused winner-take-most. Generated views paint the same surface point
differently, so a single camera can win by a few per cent of quality while two others agree on a
different colour (red feather strokes painted in one view only). Each fallback observation's quality is
therefore scaled by the share of the total observation quality that corroborates its colour
(``agreement_quality``). A clearly better camera is unaffected; only near-ties between disagreeing
cameras are decided by the majority. Observations are gathered first and fused afterwards.
"""
import os

import numpy as np
from . import fastbake as fb
from .anchor_diagnostics import AnchorDiagnostics

# Colour distance is measured in sqrt(linear RGB): roughly perceptual, no palette or asset rule.
AGREE_FULL = 0.10      # colours closer than this corroborate each other completely
AGREE_NONE = 0.22      # colours further apart than this do not corroborate at all
AGREE_POWER = 1.5      # exponent of the corroborating share applied to the quality
OBS_MIN_Q = 0.01       # observations weaker than this cannot affect any texel (relative weight q**64)


def _clock(device, torch):
    """Wall clock after the queued GPU work finished, so stage times are not attributed to the next sync."""
    import time
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


HANDOVER_NONE = 0.60   # front quality / best other camera's quality: below this the better camera paints the texel
HANDOVER_FULL = 0.80   # above this the front view keeps the texel (smooth hand-over in between)


def anchor_weight(q, fallback_q, torch, handover_q=None):
    """Weight of the front view. fallback_q: best quality of any other camera; handover_q: best quality of the cameras
    the front view may hand a texel to (default: all of them)."""
    t = ((q - 0.12) / 0.18).clamp(0, 1)
    a = t * t * (3 - 2 * t)
    if os.environ.get('UTX_ANCHOR_HANDOVER', '1') != '0':
        # The front view seeing a surface at a steep angle (chest beside a braid, the face of a tilted button) spreads a
        # few pixels over a large area, while a 45-degree camera may see it squarely. Measured on a wizard mesh: front
        # quality 0.33-0.56 against 0.93-0.99 for the 45-degree front camera on 6 % of the texels the front view owned.
        other = fallback_q if handover_q is None else handover_q
        r = ((q / other.clamp_min(1e-6) - HANDOVER_NONE) / (HANDOVER_FULL - HANDOVER_NONE)).clamp(0, 1)
        a = a * (r * r * (3 - 2 * r))
    return torch.where(fallback_q < fb.Q_LO, (q >= fb.Q_LO).float(), a)


def handover_views(views):
    """Cameras that may take a texel from the front view: not the 0-degree top/bottom ones, whose FLUX views are the
    weakest (the bottom view is mostly near-black); they still fill what no other camera sees."""
    return [vi for vi in range(1, len(views)) if not views[vi].label.endswith((":top", ":bottom"))]


def agreement_quality(colour, q, torch):
    """Scale every view's quality by the share of observation quality that agrees with its colour.

    colour (V,m,3) linear RGB, q (V,m) -> (V,m). A texel seen by one view is unchanged (share 1).
    """
    f = colour.clamp_min(0).sqrt()
    observed = torch.where(q >= fb.Q_LO, q, torch.zeros_like(q))
    support = torch.empty_like(observed)
    for j in range(f.shape[0]):
        t = (((f - f[j:j + 1]).norm(dim=2) - AGREE_FULL) / (AGREE_NONE - AGREE_FULL)).clamp_(0, 1)
        support[j] = (observed * (1 - t * t * (3 - 2 * t))).sum(0)
    share = support / observed.sum(0).clamp_min(1e-20)
    return q * share.clamp_(0, 1).pow(AGREE_POWER)


GPU_RESERVE = 1536 << 20   # free device memory that must remain after a view's observations are stored on it


class _Observations:
    """Sparse per-view observations (texel index, gain-matched colour, quality) in full precision.

    Only texels a view actually sees are stored (about 1.8 views per texel, 20 bytes each). A view's
    observations stay on the compute device while that leaves GPU_RESERVE free, and otherwise move to host
    memory, so the bake adapts to the card instead of assuming one.
    """

    def __init__(self, views, device, torch):
        self.torch, self.device = torch, device
        self.index = [None] * views
        self.colour = [None] * views
        self.quality = [None] * views
        self._parts = [[] for _ in range(views)]

    def add(self, view, start, c, q):
        keep = (q > OBS_MIN_Q).nonzero(as_tuple=True)[0]
        self._parts[view].append(((keep + start).int(), c[keep], q[keep]))

    def seal(self, view):
        torch = self.torch
        parts, self._parts[view] = self._parts[view], []
        index = torch.cat([p[0] for p in parts])
        colour = torch.cat([p[1] for p in parts])
        quality = torch.cat([p[2] for p in parts])
        del parts
        if torch.device(self.device).type == 'cuda':
            need = index.numel() * 20
            if torch.cuda.mem_get_info(self.device)[0] - need < GPU_RESERVE:
                index, colour, quality = index.cpu(), colour.cpu(), quality.cpu()
        self.index[view], self.colour[view], self.quality[view] = index, colour, quality

    def dense(self, start, stop):
        torch, device = self.torch, self.device
        views = len(self.index)
        c = torch.zeros((views, stop - start, 3), device=device)
        q = torch.zeros((views, stop - start), device=device)
        bounds = torch.tensor([start, stop], dtype=torch.int32)
        for v in range(views):
            index = self.index[v]
            lo, hi = torch.searchsorted(index, bounds.to(index.device)).tolist()
            if hi > lo:
                pos = index[lo:hi].to(device).long() - start
                c[v, pos] = self.colour[v][lo:hi].to(device)
                q[v, pos] = self.quality[v][lo:hi].to(device)
        return c, q


def bake(surface, collection, device, completion=True, timer=None, exact_support=False,edge_margin=0,fast=False):
    from .sharpbake import attach_source_base, proxy_view
    from .enhance import restore_original
    torch = fb._torch()
    timer = timer or fb.StageTimer()
    views = fb.collect_views(collection)
    geo = fb.texel_geometry(surface, device, analytic_footprint=True)
    n = len(surface.raster.texel)
    full = torch.zeros((n, 3), device=device)
    weight = torch.zeros(n, device=device)
    best = torch.zeros(n, device=device)     # strongest real quality of any fallback observation
    best_w = torch.zeros(n, device=device)   # same after the agreement scaling (reference of the weights)
    counts = torch.zeros(n, dtype=torch.int16, device=device)
    consensus = os.environ.get('UTX_CONSENSUS', '1') != '0'
    diagnostics = AnchorDiagnostics(n, len(views), device, torch)
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
    observations = _Observations(len(views), device, torch)

    def add_extra(c, q, positions):
        """Secondary (overlapping-UV) claims: fused into their texel with the raw quality."""
        unique = torch.unique(positions)
        old = best_w[unique].clone()
        best_w.scatter_reduce_(0, positions, q, reduce='amax', include_self=True)
        best.scatter_reduce_(0, positions, q, reduce='amax', include_self=True)
        scale = (old / best_w[unique].clamp_min(1e-20)).pow(64)
        full[unique] *= scale[:, None]
        weight[unique] *= scale
        w = (q / best_w[positions].clamp_min(1e-20)).pow(64)
        full.index_add_(0, positions, c * w[:, None])
        weight.index_add_(0, positions, w)

    def add_fallback(vi, c, q_real, q_weight, s, e):
        old = best_w[s:e]
        new = torch.maximum(old, q_weight)
        diagnostics.fallback_winner(vi, s, e, q_weight > old)
        scale = (old / new.clamp_min(1e-20)).pow(64)
        w = (q_weight / new.clamp_min(1e-20)).pow(64)
        full[s:e] = full[s:e] * scale[:, None] + c * w[:, None]
        weight[s:e] = weight[s:e] * scale + w
        best_w[s:e] = new
        best[s:e] = torch.maximum(best[s:e], q_real)

    for vi in list(range(1, len(views))) + [0]:
        fb._interrupt()
        if fast and torch.device(device).type=='cuda':
            # Fast has no edge-analysis pass to release allocator scratch.
            # Drop unused previous-view blocks before the next large image;
            # live tensors and all pixel arithmetic remain unchanged.
            torch.cuda.empty_cache()
        evidence=edge_margin>=8
        repair_margin=(6 if evidence else edge_margin) if vi else 0
        t_start = _clock(device, torch)
        pv = attach_source_base(fb.prepare_view(views[vi], 'base_color', device, complete_mips=True, stream=True, surface_guard=not fast,edge_margin=repair_margin,
                                              front_protection=views[0] if edge_margin>=6 and vi else None,
                                              cross_views=views if evidence and vi else None,
                                              cross_gains={v.label:gains[i] for i,v in enumerate(views)} if evidence and vi else None,
                                              colour_guard=evidence,colour_views=views if evidence else None))
        pv.exact_filter_support = bool(exact_support)
        pv.geometric_grazing_quality = bool(vi and repair_margin>0)
        gain = torch.tensor(gains[vi], device=device)
        t_prepared = _clock(device, torch)
        used = 0
        # Secondary UV claims receive no unrelated primary-camera preference.
        for s in range(n, len(geo.P), chunk):
            e = min(len(geo.P), s + chunk)
            c, q, _ = fb._project(geo.slice(s, e), pv, torch, source_detail=True)
            add_extra(c * gain, q, geo.extra_pos[s - n:e - n])
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            c, q, _ = fb._project(geo.slice(s, e), pv, torch, source_detail=True)
            observations.add(vi, s, c * gain, q)
            diagnostics.view_quality(vi, s, e, q)
            counts[s:e] += (q >= fb.Q_LO).to(torch.int16)
            used += int((q >= fb.Q_LO).sum())
        observations.seal(vi)
        t_done = _clock(device, torch)
        per.append(dict(view=pv.label, prepare_seconds=round(t_prepared - t_start, 3), projection_seconds=round(t_done - t_prepared, 3),
                        texels_used=used, fully_anchored_texels=0, guard_candidates=getattr(pv, 'guard_candidates', 0), guard_rejected=getattr(pv, 'guard_rejected', 0),source_edge_excluded_pixels=pv.source_edge_excluded_pixels,source_confirmation=pv.source_confirmation,source_colour_guard_pixels=pv.source_colour_guard_pixels,source_colour_consensus=pv.source_colour_consensus))
        del pv
    timer.lap('base_color:anchor_projection')
    anchored = reweighted = 0
    eligible = handover_views(views)
    for s in range(0, n, chunk):
        fb._interrupt()
        e = min(n, s + chunk)
        c, q = observations.dense(s, e)
        q_weight = agreement_quality(c, q, torch) if consensus else q
        for vi in range(1, len(views)):
            add_fallback(vi, c[vi], q[vi], q_weight[vi], s, e)
        full[s:e] /= weight[s:e].clamp_min(1e-20)[:, None]
        # A camera takes a texel from the front view only with a colour the other cameras corroborate (agreement-weighted
        # quality): a 45-degree view whose beard edge shows robe blue must not paint the beard.
        a = anchor_weight(q[0], best[s:e], torch, q_weight[eligible].amax(0) if eligible else torch.zeros_like(q[0]))
        diagnostics.anchor_step(s, e, a, q[0], best[s:e], q_weight[1:].amax(0))
        diagnostics.probe_texels(s, e, geo.P[s:e], c, q, q_weight, a, full[s:e])
        full[s:e] = c[0] * a[:, None] + full[s:e] * (1 - a[:, None])
        best[s:e] = torch.maximum(best[s:e], q[0])
        anchored += int((a >= 0.999999).sum())
        if consensus:
            reweighted += int(((q_weight[1:] < 0.9 * q[1:]) & (q[1:] >= fb.Q_LO)).any(0).sum())
    per[-1]['fully_anchored_texels'] = anchored
    del observations
    colour = np.nan_to_num(full.cpu().numpy(), nan=0, posinf=1, neginf=0)
    q = best.cpu().numpy()
    counts = counts.cpu().numpy()
    diagnostics.save(q, counts, surface.raster.texel)
    del geo, full, weight, best, best_w
    timer.lap('base_color:anchor_fusion')
    alpha = fb._smoothstep(q, fb.Q_LO, fb.Q_HI)
    area = surface.texel_area.astype(np.float64)
    stats = dict(texels=n, observed_strong_fraction=float((alpha >= 1 - 1e-06).mean()), observed_any_fraction=float((q >= fb.Q_LO).mean()), unobserved_fraction=float((q < fb.Q_LO).mean()), observed_area_fraction=float((area * alpha).sum() / max(area.sum(), 1e-30)), mean_views_per_observed_texel=float(counts[q >= fb.Q_LO].mean()) if (q >= fb.Q_LO).any() else 0, fusion=dict(mode='sharp_anchor_v2' if exact_support else 'sharp_anchor', anchor_view=views[0].label, anchor_full_quality=0.3, anchor_transition_start=0.12, anchor_colour_unchanged=True, source_images_warped=False, filter_support='exact mip/cubic texel bounds' if exact_support else 'rc3 radius estimate'), views=per, view_gains_rgb=gains.tolist(), harmonisation=gain_report)
    stats['fusion']['fallback_agreement'] = dict(enabled=consensus, colour_full=AGREE_FULL, colour_none=AGREE_NONE,
                                                 power=AGREE_POWER, texels_reweighted=reweighted)
    stats['fusion']['source_edge_margin_original_pixels']=edge_margin
    if fast:stats['fusion'].update(mode='Fast',surface_reconstruction='direct masked source filtering')
    if edge_margin>0:
        stats['fusion'].update(mode='sharp_anchor_v3',edge_repair='isolated chromatic source outliers near geometry edges',
                               geometric_grazing_quality=True,source_images_warped=False,anchor_sampling_unchanged=True,
                               repair_scope='fallback views only; preserve trusted view0',
                               edge_vram_mode='fast' if os.environ.get('UTX_EDGE_VRAM_MODE','strict')=='fast' else 'strict')
        if edge_margin>=6:
            stats['fusion'].update(mode='sharp_anchor_v4',edge_repair='multiscale chromatic outliers plus locally unsupported luminance residue near geometry edges',
                                   source_context_radii_original_pixels=[4,8,16],edge_candidate_radius_original_pixels=6,
                                   rejection_halo_original_pixels=5)
            stats['fusion']['front_visible_fallback']='retain v3 exclusion within conservatively reprojected front depth support'
        if edge_margin>=8:
            stats['fusion'].update(mode='Quality',
                edge_repair='geometry-scoped source rejection with independent two-camera photometric confirmation',
                source_edge_margin_original_pixels=6,
                rejected_source_confidence_zero=True,
                photometric_consensus=dict(minimum_alternative_cameras=2,maximum_direction_dot=.98,
                    alternative_rgb_distance=.12,relative_alternative_distance=.35,alternative_distance_cap=.22,
                    minimum_source_rgb_distance=.20,reconstruction_support_original_pixels=1,
                    source_support_rgb_distance=.14,original_source_pixels=True),
                front_visible_fallback='local protection plus two-camera confirmation; primary view0 unchanged')
            stats['fusion'].update(anchor_sampling_unchanged=True,anchor_colour_unchanged=False,
                sr_colour_constraint='same-camera depth/normal-matched chroma envelope in geometric-edge band; preserve enhanced mean light',
                primary_camera_policy='anchor weights/sampler unchanged; constrain unsupported enhanced chroma near geometry boundaries')
    if completion:
        values, stats['completion'] = fb.complete_texels(surface, colour, alpha, device)
    else:
        values = np.where((q >= fb.Q_LO)[:, None], colour, 0)
    timer.lap('base_color:completion')
    return (np.clip(fb.linear_to_srgb_np(values), 0, 1), alpha, stats)
