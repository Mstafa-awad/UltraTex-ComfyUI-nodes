"""Keep the original six-camera bake; additional cameras paint only its holes."""
import numpy as np
from . import fastbake as fb
from .contracts import ViewCollection


def bake(surface, collection, device, completion=True, timer=None, detail=False, semantic='base_color'):
    timer = timer or fb.StageTimer()
    class PartTimer:
        def __init__(self, prefix): self.prefix = prefix
        def lap(self, label): timer.lap(self.prefix + ':' + label)
    primary = ViewCollection((collection.sets[0],))
    # Preserve the working main-rig colour/detail reconstruction, including
    # its transition treatment. Extras never enter its gains or vertex field.
    base, confidence, main_stats = fb.bake_branch(surface, primary, semantic, device,
        completion=completion, harmonize=semantic=='base_color', timer=PartTimer('main'),
        detail=detail,binary_coverage=True)
    main_seen = confidence > 0
    seen = main_seen.copy()
    extra_stats = None
    supplied = 0
    if len(collection.sets) > 1:
        additional = ViewCollection(collection.sets[1:])
        extra, extra_confidence, extra_stats = fb.bake_branch(surface, additional, semantic, device,
            completion=False, harmonize=False, timer=PartTimer('additional'),detail=detail,binary_coverage=True)
        fill = ~main_seen & (extra_confidence > 0)
        base[fill] = extra[fill]
        supplied = int(fill.sum())
        seen |= fill
        del extra, extra_confidence, fill
    if completion:
        linear, fill_stats = fb.complete_texels(surface, fb.srgb_to_linear_np(base) if semantic=='base_color' else base,
                                               seen.astype(np.float32), device)
        # Copy only real holes back: even sRGB roundtrip error must not change
        # observed main pixels when nonparticipating images are added.
        missing = ~seen
        base[missing] = fb.linear_to_srgb_np(linear[missing]) if semantic=='base_color' else linear[missing]
        del linear
    else:
        base[~seen] = 0
        fill_stats = {'completed_texels': 0}
    area = surface.texel_area.astype(np.float64)
    timer.lap(semantic+':fill_remaining_holes')
    stats = dict(main_stats)
    stats.update(observed_strong_fraction=float(seen.mean()),observed_any_fraction=float(seen.mean()),
        unobserved_fraction=float((~seen).mean()),
        observed_area_fraction=float(area[seen].sum()/max(area.sum(),1e-30)),
        main_rig=main_stats,additional_rigs=extra_stats,completion=fill_stats,
        fusion=dict(mode='original_main_rig_with_gap_fill',main_camera_count=len(fb.collect_views(primary)),
            additional_camera_count=sum(len(v.plan.active_indices) for v in collection.sets[1:]),
            main_reconstruction='original v2.2 fine-detail/two-band bake' if detail else 'original v2.2 direct/two-band bake',
            main_covered_texels=int(main_seen.sum()),additional_texels_supplied=supplied,
            additional_overwrite_main_texels=0,additional_views='fill main-rig holes only',
            additional_views_influence_main_gains=False,additional_views_influence_main_completion=False,
            material_repaint=False))
    return np.clip(base,0,1),seen.astype(np.float32),stats
