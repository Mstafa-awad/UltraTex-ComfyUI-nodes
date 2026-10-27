"""One-way colour calibration: fill cameras never recolour protected sources."""
import numpy as np


def fit_gains(colour, quality, tiers, minimum_samples=64, limit=1.15):
    """Fit a constant RGB gain to earlier-tier overlap, with no reverse influence.

    Front/back retain their exact colour. Side cameras match only front/back;
    extra cameras match the already calibrated main rig. No image is averaged
    into the output, and adding/changing extras cannot affect main-rig gains.
    Spatial pattern differences are deliberately not treated as exposure.
    """
    colour = np.asarray(colour, np.float32)
    quality = np.asarray(quality, np.float32)
    tiers = np.asarray(tiers)
    if colour.shape != (*quality.shape, 3) or tiers.shape != (len(colour),):
        raise ValueError('Colour, quality and camera tiers disagree')
    gains = np.ones((len(colour), 3), np.float32)
    owner = np.full(quality.shape[1], -1, np.int32)
    target = np.zeros((quality.shape[1], 3), np.float32)
    target_q = np.zeros(quality.shape[1], np.float32)
    evidence = []
    for tier in sorted(set(tiers.tolist())):
        indices = np.flatnonzero(tiers == tier)
        # All cameras in this tier see only preceding tiers as calibration data.
        for vi in indices:
            mask = ((owner >= 0) & (quality[vi] > .2) & (target_q > .2)
                    & (colour[vi].mean(1) > .01) & (target.mean(1) > .01))
            samples = int(mask.sum())
            if samples >= minimum_samples:
                delta = np.log(np.maximum(target[mask], 1e-3)) - np.log(np.maximum(colour[vi, mask], 1e-3))
                weights = np.minimum(quality[vi, mask], target_q[mask])[:, None]
                value = np.median(delta, axis=0)
                for _ in range(3):
                    robust = weights / (1 + ((delta - value) / .1) ** 2)
                    value = (robust * delta).sum(0) / np.maximum(robust.sum(0), 1e-8)
                gains[vi] = np.exp(np.clip(value, -np.log(limit), np.log(limit)))
            evidence.append(dict(view=int(vi), tier=int(tier), overlap_samples=samples))
        q = quality[indices]
        chosen = q.argmax(0)
        best = q[chosen, np.arange(q.shape[1])]
        fill = (owner < 0) & (best >= .06)
        selected = indices[chosen[fill]]
        owner[fill] = selected
        target_q[fill] = best[fill]
        target[fill] = colour[selected, np.flatnonzero(fill)] * gains[selected]
    return gains, dict(mode='earlier-tier overlap only', protected_source_gains_exact=True,
                       extra_views_influence_main=False, gain_limit=limit, views=evidence)
