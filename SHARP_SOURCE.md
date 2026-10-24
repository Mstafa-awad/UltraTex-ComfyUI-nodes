> Historical release note. Current bake presets are Fast, Quality and Quality + edge repair; see README.md.

# Source-preserving bake TEST — 2.2.0rc2

In UTX3 choose `upscale_mode=sharp_source`. All three previous modes remain available; the default remains `fast_detail` for existing workflows. No node socket order changed.

For RealESR use UTX2.5 `Comfy upscale model`, connect the existing `realesr-general-x4v3.pth` loader, and select `target=Native model scale`. This keeps the checkpoint's actual output pixels, pads the rectangular result without stretching it, and avoids the second resize to a fixed 8K canvas. It does not select or download a model automatically. Original views and cameras remain available for rollback. Native scale requires an integer-scale Comfy upscale model; VOSR2 uses its existing target choices.

Choose the **final exported texture resolution in UTX3**. UTX2.5's 8192 setting changes source views only. Start with a 4096 atlas. The optional 8192 atlas requires `sharp_source`, more RAM and more baking time; it is intended for close-up review. It keeps the same UVs and material geometry. The 8K path drops other cached UltraTex surface rasters before preparing its own, checks system-RAM headroom and retains the existing CUDA/CPU budget guard and OOM fallback.

The new mode separates detail at sigma 8 ORIGINAL camera pixels, adjusted for each exact crop/resize affine. It combines full source samples with a narrow continuous quality preference (relative q**64) and mixes only a small coarse colour base. Colour adjustment is multiplicative and limited to +/-10%, so the correction cannot lift black pixels into grey/brown. Projection retains the original depth test; source images are not warped or redrawn. Differential UV footprints and complete mask-normalised mips control minification. Cubic reconstruction is bounded by observed neighbouring pixels and requires valid 4x4 support.

The previous three-edge mesh low-pass is not used in this mode. On very dense meshes, a few graph edges are too small a physical radius to separate broad colour from misregistered iris/marking details. A one-pixel band in an enhanced 8K canvas also protects a different feature scale from a one-pixel band before enhancement.

Preparation streams bounded rows into the same premultiplied GPU mip pyramid. A complete real 8K view's mip arrays and coarse colour base matched the unstreamed preparation bit-for-bit in the GPU validation. Exposure matching reads preserved original views; final detail and coarse colour still come from the selected enhanced images. MR stays on its original scalar-data branch.

This mode reduces averaging of conflicting source features; it cannot make conflicting generated views identical or create faithful details absent from them. Narrower handovers can expose a source mismatch. It is not MVS-Texturing, optical-flow registration, UV repacking or automatic facial retouching. The diagnostic hard-winner experiment was rejected for seams and uncompleted black regions. Inspect the actual completed neck/collar, both eyes, back and UV seams before accepting an asset.

No additional dependency or checkpoint is introduced. New code uses the existing NumPy/SciPy/PyTorch stack under this project's existing GPL-3.0-only distribution terms. Commercial asset production is not prohibited by the added code; source redistribution retains the existing GPL obligations. RealESR checkpoint terms remain separate. The tested checkpoint came from the official Real-ESRGAN BSD-3-Clause project release; no weights are bundled.

Install: close ComfyUI, back up the current `comfyui_ultratex_modular` folder outside `custom_nodes`, replace it with the single folder from the ZIP, and restart. Keep the generation and MR wiring unchanged. Use the bundled latest-ten-view presets as examples. Per-bake rollback: choose `fast_detail`; enhancement rollback: disable UTX2.5 or use `original_views`. Full rollback: restore the preserved v2.2.0rc1 ZIP/folder. Installed runtime/model files were not changed by this research.
