# View 0 anchor TEST, v2.2.0rc3

In UTX 3 choose `upscale_mode=sharp_anchor` and `completion=surface_fill`.
Keep your RealESR model in UTX 2.5, `Native model scale`, strength 1, geometry
mask and crop enabled. The final exported atlas size belongs to UTX 3: choose
4096 for a preview or 8192 for the measured close-up result.

This mode deliberately preserves the first active view (normally rig0/front).
Where its geometry-tested quality is at least 0.30, its sampled RGB is retained
without exposure correction or blending with other generated views. Quality
0.12–0.30 makes a smooth transition to the best available other view. An only
available front observation remains usable from quality 0.06. These numbers are
projection quality thresholds, not semantic confidence or accuracy percentages.

Every candidate colour tap near a depth/normal boundary is tested against the
target triangle plane and surface normal. Invalid coverage is not painted black:
other cameras and the existing mesh-surface completion supply the gaps. This
keeps view 0's cleaner facial landmarks without treating a raw single-camera
projection as a complete texture. Black markings already present in a valid
front observation can remain. No automatic method here knows the intended
semantic colour of every surface.

No mesh, UV, camera fit, model, LoRA, generation setting, SLA ratio or environment
is changed. MR uses the retained scalar-data path. No new dependency, learned
model, API or model download is required. The code remains GPL-3.0-only;
commercial use is allowed under that license's obligations. Existing third-party
notices are retained. This release does not audit the user's existing models.

`sharp_coherent` is an additional experimental graph-cut option, not the
recommended result after the user's view-0 comparison. Its CPU guide pass and
bounded log-colour correction can resolve local source conflicts but add cold
bake time and do not ensure the preferred character appearance. Its exact guide
arrays are cached through the existing UltraTex Store. Clearing that cache only
forces the guide to be recalculated; it does not change camera fitting.

Rollback per bake: choose `sharp_source` (the accepted rc2 calculation remains
bitwise verified), or any older retained mode. Full rollback: close ComfyUI,
replace the plugin folder with the supplied rc2 rollback ZIP, then restart.
Keep only one UltraTex folder active under custom_nodes. No installation was
performed on your live copy during this research.

See the engineering report for the measured speed/memory tradeoff, benchmark
definitions, source-scene limitations, and before/after images. This TEST release
does not claim 99% semantic accuracy on arbitrary characters or assets.
