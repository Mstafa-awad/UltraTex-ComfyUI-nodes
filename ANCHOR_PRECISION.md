> Historical release note. Current bake presets are Fast, Quality and Quality + edge repair; see README.md.

# Anchor Precision TEST, 2.2.0rc4

Select `sharp_anchor_v2` in UTX 3 to test the corrected geometry guard. Keep
RealESR, atlas size, camera fitting and all generation settings as before.
`sharp_anchor` retains the accepted rc3 calculation for immediate rollback.

The fixed bug concerns filter support. A coarse mip sample reads complete blocks
of original image pixels. The rc3 edge query estimated a smaller radius and
could miss a nearby occluding surface inside the actual mip support. The new
helper derives bounds from the two sampled mip blocks and the cubic 4x4 window,
maps them through the exact enhancement-crop transform to the original depth
buffer, and checks a covering edge-pyramid region. Odd-sized buffers which lack
a sufficiently coarse edge level use the guarded path conservatively.

The front anchor's exposure, preference thresholds, camera matrices and texture
images are unchanged. MR remains on its original data-map path. The fix does not
remove legitimate black detail, edit painted features, remesh, unwrap, or switch
to nearest-neighbor display to hide interpolation. It introduces no checkpoint.
The CPU thread controller uses threadpoolctl 3.6 (BSD-3-Clause), already present
in the tested ComfyUI runtime. Existing GPL-3.0-only source and notices apply.

All available logical processors are enabled for PyTorch, BLAS and OpenMP
CPU math pools (16 on the tested laptop). This is a process-wide setting, so
other ComfyUI CPU nodes can also benefit. Serial raster/sparse implementations
remain serial; enabling 16 threads does not promise 100% CPU utilisation.
SciPy nearest-neighbour queries in completion and normal repair request all
workers explicitly. The optional standalone converter also defaults to all
logical cores, with its explicit --threads override retained.

Large UV raster candidate chunks run as independent ordered thread jobs. The
existing aggregate scratch allowance is divided among workers rather than
multiplied by their count. Stable ownership of overlaps is preserved. Oversized
individual triangles keep the serial path. Large harmonic fills solve their
independent RGB columns concurrently using one shared sparse matrix; nested
BLAS threads are limited during these jobs to avoid oversubscription. Set
`UTX_CPU_PARALLEL=0` before starting ComfyUI to disable these independent jobs.
Native CPU math pools still use all cores; install rc3 for the previous policy.

At the user's request, predictive RAM/VRAM refusals and automatic baker CPU
redirection/reruns are removed. Real allocation errors still propagate. Cache
reclamation, model offload and memory-efficient computation remain active.
Geometry visibility, numerical checks, UV validation and disk-space/budget
checks are retained. To restore the previous memory policies install rc3.

Readable code and an isolated support helper make the algorithm easier to audit;
formatting alone is not credited with better image quality. The confirmed
regression is geometric cross-surface filtering. Generated paint errors and
roughness/geometry reflections are separate, and zero bleed on every character
is not proven. Read the engineering report for GPU versus CPU test scope.

The optional debug workflow retains the user's MemoryCleaner gates. That node
unloads models and runs garbage collection; it does not delete source-view files.
UTX 4 is a different disk-cache cleaner. The debug workflow bypasses UTX 4 so
source observations survive long enough for an exact artifact investigation.
