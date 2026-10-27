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


def source_tiers(views):
    """Main front/back, remaining main cameras, then additional cameras."""
    tiers = []
    for view in views:
        rig, _, name = view.label.partition(':')
        tiers.append(0 if rig == 'rig0' and name in ('front', 'back') else
                     1 if rig == 'rig0' else 2)
    return tiers


def select_sources(colour, q, tiers, torch):
    """One sharp source per texel; later cameras fill holes only.

    Visibility/filter validity is already encoded in q. Once an earlier tier
    supplies a usable observation, a later tier cannot replace it merely by
    having a better angle. This also protects weak but valid main observations.
    """
    tier = torch.tensor(tiers, device=q.device)[:, None]
    missing = len(tiers) + 3
    chosen_tier = torch.where(q >= fb.Q_LO, tier, missing).amin(0)
    allowed = torch.where((tier == chosen_tier) & (q >= fb.Q_LO), q, 0.)
    quality, owner = allowed.max(0)
    index = torch.arange(q.shape[1], device=q.device)
    result = colour[owner, index]
    owner = torch.where(quality >= fb.Q_LO, owner, -1)
    result = torch.where((owner >= 0)[:, None], result, 0.)
    return result, quality, owner, allowed


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
    if os.environ.get('UTX_ANCHOR_HANDOVER', '0') != '0':
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


HELD_FROM = 0.35   # first-rig quality where the guided pass starts holding a 45-degree pixel (guidance.reproject_atlas)


def guided_split(views, collection):
    """Indices for opt-in novelty weighting; normal bridge fusion returns ([], [])."""
    guided_sets = {si for si, vs in enumerate(collection.sets) if getattr(vs, "guided", False)}
    # The guided views bridge the different micro-textures painted by the main
    # rig. Suppressing their held regions (2.3.9) exposes a patchwork of sharp
    # but incompatible grain. Keep the proven bridge fusion by default; retain
    # novelty-only fusion as an explicit diagnostic experiment.
    if not guided_sets or os.environ.get('UTX_GUIDED_NOVELTY', '0') == '0':
        return [], []
    rig = [int(v.label.split(":")[0][3:]) for v in views]
    return [i for i, r in enumerate(rig) if r not in guided_sets], [i for i, r in enumerate(rig) if r in guided_sets]


def novel_quality(q, first, guided, torch):
    """Quality of the guided views counted only where they carry new information.

    Where the first rig saw a texel well, the guided 45-degree pass held its pixel to a re-render of the first-pass
    texture (a 2048 atlas, then a VAE round trip): a blurred copy of the first rig. With the plain quality those copies
    beat the sharp 0-degree pixels on every surface facing a 45-degree camera (back, sides), because the fusion is
    winner-take-most on viewing angle. Scaling them by (1 - hold strength) keeps them where they painted something new
    (under the beard, inside the sleeves) and lets the original views paint the rest.
    """
    if not guided or not first:
        return q
    held = ((q[first].amax(0) - HELD_FROM) / (1 - HELD_FROM)).clamp(0, 1)
    out = q.clone()
    out[guided] *= 1 - held
    return out


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


def bake(surface, collection, device, completion=True, timer=None, exact_support=False,edge_margin=0,fast=False,source_priority=False):
    from .sharpbake import attach_source_base, proxy_view
    from .enhance import restore_original
    torch = fb._torch()
    timer = timer or fb.StageTimer()
    views = fb.collect_views(collection)
    tiers = source_tiers(views)
    geo = fb.texel_geometry(surface, device, analytic_footprint=True)
    n = len(surface.raster.texel)
    full = torch.zeros((n, 3), device=device)
    weight = torch.zeros(n, device=device)
    best = torch.zeros(n, device=device)     # strongest real quality of any fallback observation
    best_w = torch.zeros(n, device=device)   # same after the agreement scaling (reference of the weights)
    counts = torch.zeros(n, dtype=torch.int16, device=device)
    # Restore the working source ownership as one policy: a stable front anchor
    # plus ordinary guided candidates, without colour voting changing the owner.
    consensus = not source_priority and os.environ.get('UTX_CONSENSUS', '0') != '0'
    diagnostics = AnchorDiagnostics(n, len(views), device, torch)
    timer.lap('base_color:anchor_geometry')
    originals = fb.collect_views(restore_original(collection))
    originals = [proxy_view(v) if v.image_ref.shape[0] > 2048 else v for v in originals]
    gains, gain_report = fb.estimate_gains(geo, originals, device, complete_mips=True,
                                         source_tiers=tiers if source_priority else None)
    # Keep the anchor exposure exact; harmonize only fallback cameras.
    gains = np.clip(gains / np.maximum(gains[0], 1e-08), 1 / 1.15, 1.15)
    gains[0] = 1
    timer.lap('base_color:anchor_colour_matching')
    per = []
    chunk = 1 << 18
    observations = _Observations(len(views), device, torch)
    secondary_tier = (torch.full((n,), len(views)+3, dtype=torch.int16, device=device)
                      if source_priority and geo.extra_pos is not None else None)

    def add_extra(c, q, positions, vi):
        """Secondary (overlapping-UV) claims: fused into their texel with the raw quality."""
        unique = torch.unique(positions)
        if source_priority:
            # An overlapping UV claim may fill a hole; never let it repaint an
            # observed primary surface. Keep one sample even for repeated IDs.
            unique, inverse = torch.unique(positions, return_inverse=True)
            maximum = torch.zeros(len(unique), device=device)
            maximum.scatter_reduce_(0, inverse, q, reduce='amax', include_self=True)
            earlier = tiers[vi] < secondary_tier[positions]
            same = tiers[vi] == secondary_tier[positions]
            candidate = ((q == maximum[inverse]) & (q >= fb.Q_LO)
                         & (earlier | (same & (q > best[positions]))))
            order = torch.arange(len(q), device=device)
            picked = torch.full((len(unique),), len(q), dtype=torch.int64, device=device)
            picked.scatter_reduce_(0, inverse, torch.where(candidate, order, len(q)), reduce='amin', include_self=True)
            valid = picked < len(q)
            target = unique[valid]
            full[target] = c[picked[valid]]
            weight[target] = 1
            best[target] = maximum[valid]
            secondary_tier[target] = tiers[vi]
            return
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
            add_extra(c * gain, q, geo.extra_pos[s - n:e - n], vi)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            c, q, _ = fb._project(geo.slice(s, e), pv, torch, source_detail=True)
            observations.add(vi, s, c * gain, q)
            diagnostics.view_quality(vi, s, e, q)
            counts[s:e] += (q >= fb.Q_LO).to(torch.int16)
            used += int((q >= fb.Q_LO).sum())
        observations.seal(vi)
        t_done = _clock(device, torch)
        per.append(dict(view=pv.label, image_px=pv.S, depth_px=pv.R, enhanced=views[vi].transform is not None,
                        prepare_seconds=round(t_prepared - t_start, 3), projection_seconds=round(t_done - t_prepared, 3),
                        texels_used=used, fully_anchored_texels=0, guard_candidates=getattr(pv, 'guard_candidates', 0), guard_rejected=getattr(pv, 'guard_rejected', 0),source_edge_excluded_pixels=pv.source_edge_excluded_pixels,source_confirmation=pv.source_confirmation,source_colour_guard_pixels=pv.source_colour_guard_pixels,source_colour_consensus=pv.source_colour_consensus))
        del pv
    timer.lap('base_color:anchor_projection')
    anchored = reweighted = 0
    eligible = handover_views(views)
    first, guided = guided_split(views, collection)
    supplied = torch.zeros(len(views), dtype=torch.int64, device=device)
    protected_main = extra_overwrites = 0
    extra_indices = torch.unique(geo.extra_pos) if source_priority and geo.extra_pos is not None else None
    if extra_indices is not None:
        extra_colour, extra_quality = full[extra_indices].clone(), best[extra_indices].clone()
    for s in range(0, n, chunk):
        fb._interrupt()
        e = min(n, s + chunk)
        c, q = observations.dense(s, e)
        if source_priority:
            chosen, quality, owner, allowed = select_sources(c, q, tiers, torch)
            main = [i for i, tier in enumerate(tiers) if tier < 2]
            main_seen = (q[main] >= fb.Q_LO).any(0)
            protected_main += int(main_seen.sum())
            extra_owner = torch.zeros_like(main_seen)
            for vi, tier in enumerate(tiers):
                if tier == 2: extra_owner |= owner == vi
            extra_overwrites += int((main_seen & extra_owner).sum())
            full[s:e] = chosen
            best[s:e] = quality
            supplied += torch.bincount(owner[owner >= 0], minlength=len(views))
            for vi in range(len(views)):
                diagnostics.fallback_winner(vi, s, e, owner == vi)
            a = (owner == 0).float()
            diagnostics.anchor_step(s, e, a, q[0], q[1:].amax(0) if len(views)>1 else torch.zeros_like(q[0]), quality)
            diagnostics.probe_texels(s, e, geo.P[s:e], c, q, allowed, a, chosen)
            anchored += int((owner == 0).sum())
            continue
        qn = novel_quality(q, first, guided, torch)
        q_weight = agreement_quality(c, qn, torch) if consensus else qn
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
    if extra_indices is not None:
        missing = (best[extra_indices] < fb.Q_LO) & (extra_quality >= fb.Q_LO)
        full[extra_indices[missing]] = extra_colour[missing]
        best[extra_indices[missing]] = extra_quality[missing]
    if source_priority:
        supplied_counts = supplied.cpu().tolist()
        for row in per:
            vi = next(i for i, v in enumerate(views) if v.label == row['view'])
            row.update(source_tier=tiers[vi], texels_supplied=supplied_counts[vi])
    del observations
    colour = np.nan_to_num(full.cpu().numpy(), nan=0, posinf=1, neginf=0)
    q = best.cpu().numpy()
    counts = counts.cpu().numpy()
    diagnostics.save(q, counts, surface.raster.texel)
    del geo, full, weight, best, best_w
    timer.lap('base_color:anchor_fusion')
    # Completion must fill holes, not blend a low-frequency vertex field into
    # valid, steep-angle observations. Keep every selected source pixel intact.
    alpha = ((q >= fb.Q_LO).astype(np.float32) if source_priority else
             fb._smoothstep(q, fb.Q_LO, fb.Q_HI))
    area = surface.texel_area.astype(np.float64)
    stats = dict(texels=n, observed_strong_fraction=float((alpha >= 1 - 1e-06).mean()), observed_any_fraction=float((q >= fb.Q_LO).mean()), unobserved_fraction=float((q < fb.Q_LO).mean()), observed_area_fraction=float((area * alpha).sum() / max(area.sum(), 1e-30)), mean_views_per_observed_texel=float(counts[q >= fb.Q_LO].mean()) if (q >= fb.Q_LO).any() else 0, fusion=dict(mode='sharp_anchor_v2' if exact_support else 'sharp_anchor', anchor_view=views[0].label, anchor_full_quality=0.3, anchor_transition_start=0.12, anchor_colour_unchanged=True, source_images_warped=False, filter_support='exact mip/cubic texel bounds' if exact_support else 'rc3 radius estimate'), views=per, view_gains_rgb=gains.tolist(), harmonisation=gain_report)
    stats['fusion']['fallback_agreement'] = dict(enabled=consensus, colour_full=AGREE_FULL, colour_none=AGREE_NONE,
                                                 power=AGREE_POWER, texels_reweighted=reweighted)
    stats['fusion']['guided_views_new_only'] = dict(enabled=bool(guided), views=[views[i].label for i in guided], held_from=HELD_FROM)
    has_guided = any(getattr(vs, 'guided', False) for vs in collection.sets)
    stats['fusion']['guided_bridge_policy'] = ('novelty_only' if guided else 'consistent_with_main_rig') if has_guided else 'no_guided_rig'
    stats['fusion']['front_camera_policy'] = 'quality_handover' if os.environ.get('UTX_ANCHOR_HANDOVER', '0') != '0' else 'legacy_anchor'
    if source_priority:
        for obsolete in ('anchor_full_quality', 'anchor_transition_start', 'fallback_agreement'):
            stats['fusion'].pop(obsolete, None)
        stats['fusion'].update(mode='coverage_priority',
            source_order=['main front/back', 'remaining main cameras', 'additional cameras'],
            minimum_quality=fb.Q_LO, angle_based_handover=False,
            one_source_per_texel=True, cross_view_rgb_averaging=False,
            additional_views='fill missing main-rig coverage only',
            main_covered_texels=protected_main,
            additional_overwrite_main_texels=extra_overwrites,
            weak_observed_texels_preserved=int(((q >= fb.Q_LO) & (q < fb.Q_HI)).sum()),
            completion_scope='unobserved texels only',
            front_camera_policy='main front/back first; unchanged source detail',
            guided_bridge_policy='missing_coverage_only',
            guided_views_new_only=dict(enabled=True, views=[v.label for i,v in enumerate(views) if tiers[i]==2]))
    stats['fusion']['source_edge_margin_original_pixels']=edge_margin
    if fast:stats['fusion'].update(surface_reconstruction='direct masked source filtering')
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
