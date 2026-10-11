# UltraTex Modular v0.2 — triangle artifact and 2K workflow upgrade

## Installation

1. **Back up** your existing `ComfyUI/custom_nodes/comfyui_ultratex_modular` folder and any edited workflows. Never install two active copies of nodes with the same `UTX...` IDs.
2. Extract this ZIP, then copy its one `comfyui_ultratex_modular` folder into `ComfyUI/custom_nodes/` (replace the older folder, don't nest it).
3. Dependencies remain those of v0.1.2. From the portable ComfyUI parent folder, run (only if needed):

   `python_embeded\\python.exe -m pip install -r ComfyUI\\custom_nodes\\comfyui_ultratex_modular\\requirements.txt`

   No new model weights. The original validated Klein BASE 9B GGUF, matching UltraTex albedo LoRA, Qwen3-8B, FLUX.2 VAE and UltraTex decoder are still needed. The sparse backend continues to require working compatible Triton on Windows.
4. Restart ComfyUI. Drag one of the JSONs from **`workflows/`** onto the canvas. Select the GLB, reference PNG and model filenames (example names are placeholders).

## Which workflow?

| Workflow | What it does | Expected tradeoff |
| --- | --- | --- |
| `01_NATIVE_2048_6V_8GB.json` | 6 native 2048 views, 2048 UV atlas, 25-step default | Start here to test real 2K quality; **not guaranteed** to fit all 8GB configurations without offload |
| `02_NATIVE_2048_12V_GUIDED.json` | 6 native-2K views, project first-pass atlas into rotated six-view cameras, 16-step low-denoise continuation, 2K bake | More side coverage; takes longer and needs more system RAM and disk |
| `03_FAST_1024V_TO_2048_ATLAS.json` | 6 native-1024 views, bakes to a 2048 atlas | Faster and less VRAM; **not native 2K image detail** |
| `04_REFERENCE_BLEND_2048_6V.json` | Old-style closest-view blending/filling on the updated normals | A/B against new multiband baker, also 2048-native |

The GLB **must have valid UVs** with no stacked islands. A single static mesh is supported; see README for GLB restrictions. Changing only Bake node settings should reuse generated views. The created GLB does not remesh your original mesh.

## Changes made

* **Normal conditioning control** in Prepare Texture: `auto_smooth` (default), `preserve`, `smooth`. The input MESH is not mutated. `auto_smooth` welds coincident positions for smooth normal computation but preserves strongly divergent authored normals, so triangles from faceted input normal data are reduced without smoothing the actual geometry. Preserve is the exact original-normal reference.
* **Guided 45° views:** a first-pass bake is re-rendered through the second camera rig. Known regions constrain latent diffusion, while unobserved areas remain editable. The initial atlas's empty regions are filled only for guidance; they do not become hard preservation masks. New controls: `extra_angle_mode`, `extra_denoise`, `extra_steps`. The independent original pass remains for comparison and MR tasks.
* **Detail-preserving projection:** same UV/depth calibration, but uses frequency-separated fusion to avoid suppressing high-frequency edges and patterns with broad color averaging. Optional reference/legacy/smooth modes are retained for A/B testing; these fusion modes are **experimental and not quality-benchmarked on the user's GPU**.
* **Normal-aware hole filling:** searches nearby 3D texels but prefers similarly oriented surfaces. It won't copy a pixel from the back of a face if a compatible nearby surface exists. It remains a color estimation method, not texture synthesis for hidden geometry.
* **Projection confidence output:** the Bake Texture node has a fifth output, MASK `projection_confidence`. Use Preview Mask if needed to inspect low-evidence regions.
* **Stable cache separation:** normal and G-buffer cache identities are upgraded to prevent an obsolete normal map from being reused.
* **Preserved 8GB-oriented runtime:** foreground token dropping, sparse attention, sequential denoising, GGUF Q4, disk-backed intermediates, tiled VAE decoding, chunked transformer execution, and 2K support are unchanged rather than regressed by switching to heavier full precision.

## Verification and limits

* CPU regression tests: **33 passing**, run with `python -m unittest comfyui_ultratex_modular.tests.test_core comfyui_ultratex_modular.tests.test_quality_upgrade -q` from the parent folder.
* Workflow JSON: link graph, source types and destination types structurally validated. Syntax compiled.
* **Not verified:** actual FLUX.2 inference, actual ModernGL raster outputs (ModernGL unavailable in this build environment), actual 8GB GPU utilization, speedups, visual elimination of all artifacts, and compatibility with a live ComfyUI installation. Those require a real ComfyUI/GPU run.
* This is a **working implementation candidate**, not a measured guarantee of better visual quality or faster native 2048. Genuine geometry faceting and bad/non-overlapping UV unwraps cannot be completely repaired by shading, baking or a better sampler.
* If a face still looks triangular, test `normal_mode=preserve` vs `smooth`, then use `04_REFERENCE_BLEND_2048_6V.json` to compare the baker on identical views. Check that the projection-confidence mask is high on the face; low confidence points to missed visible texels or occlusion/depth sampling.

There are no redistributed model weights or font files in this ZIP.
