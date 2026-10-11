# Validation — 10 October 2026

## Revision 0.1.1

The current release passes 23 CPU regression tests and native ComfyUI block parity checks (maximum difference 2.38e-7 in three float32 activation variants). Ten example graphs have their required inputs/link types checked. Wrong MR/albedo selection is rejected by V3 input validation, with publisher hash precedence tested against misleading filenames. Full Q4_0 model comparisons on the actual character produced identical foreground latents across the tested memory/chunk changes (maximum absolute difference 0.0).

See [FIXES_0.1.1.md](FIXES_0.1.1.md) for the actual-asset rebake, matched three-step timings, preserved post1 fixes and limits. The local adapter filenames were found to be swapped, verified against published ModelScope hashes. Initial timing comparisons used the same MR artifact despite being labelled albedo. A 25-step run of that artifact completed denoising in 315.96 seconds (12.64s/step); sampling including group loading took 325.21 seconds. Its red material output was rejected as a base-color result. These are 1024px views, separately baked to 2048px. Real cached Qwen3-8B text was used; cold text encoding and the full GUI graph are excluded. This remains slower than the supplied other-machine FP8 log. Do not extrapolate the short-test improvement into a claim that the speed gap is solved.

Raw 0.1.1 reports are prefixed `revision_` in `validation/`. The earlier results below are retained as historical integration checks and do not supersede the new actual-asset evidence.

The subsequent verified-albedo 25-step run completed successfully: 315.92s denoising, 325.27s sampling with loading, 16.99s uncached six-view decode, 4.99s bake and 354.78s from sampling through export. It excludes initial model/text loading and condition preparation. Its purple/gold generated front and textured-mesh render were inspected. Residual holes and some generated skirt artifacts remain. The user then corrected the adapter filenames; final workflows and the current component manifest use the reverified names. Historical reports preserve their original filenames and hashes.

## Historical 0.1.0 checks

The task labels in the historical reports below were inferred from local filenames, which turned out to be swapped. The hash in each report identifies the actual artifact: `3b191517...` is albedo; `13e94f5a...` is MR. Numerical integration checks remain relevant, but these labels cannot establish albedo/MR quality.

Initial implementation tested on Windows, Python 3.12.10, PyTorch 2.8.0+cu128, an RTX 5050 Laptop GPU (8,151 MiB), and approximately 16 GB system RAM. Exact dependency versions, host revisions and local model hashes are in `component_manifest.json`.

## Passed checks

- 14 CPU unit tests: immutable array reads and incomplete-cache rejection; disk budget; geometry preservation and padded mesh counts; UV rejection; inherited material factors; scalar channel order; map metadata mismatch; explicit UV convention conversion; independent opacity; encoding reuse; canonical position IDs; preservation-mask packing; schedule boundaries; disk-backed decode/background reconstruction; portable bundle round trip/path boundary; Qwen attention-mask metadata; scoped GGUF weight reuse. Some tests exercise several related assertions.
- A mocked 20-job workset made one model-manager request, reused all completed latents without another request, recorded cancellation after three completed jobs, and resumed the remaining jobs. This is an orchestration test, not 20 real diffusion jobs.
- All 27 V3 schemas registered. Eight workflow graphs had required sockets/widgets and link types checked against actual installed schemas.
- ModernGL rendered six canonical views; a 4096 atlas rasterized in 64 tiles. Asymmetric world-coordinate colors baked in the expected UV orientation. Native MESH -> material -> GLB -> native MESH preserved tested texture values and alpha metadata. Original mesh data remained unchanged. Projection cache reuse and overlapping-UV rejection passed.
- The real foreground decoder matched 140 expected tensors, decoded successfully, and the unpatched sibling VAE output was restored exactly after switching branches.
- Both released UltraTex adapters matched all 80 expected layers. Real Q4 9B sampling ran sparse attention without a dense fallback at 512 and 2048. A combined albedo/MR workset switched branches of the shared base and decoded all 12 views.
- The real Qwen3-8B Q6_K text encoder produced finite `(1,512,12288)` empty-prompt conditioning through UTX Text Encode, including native `attention_mask` metadata. Text encoding was tested separately from diffusion.

## Recorded timings, not production benchmarks

The diffusion fixture is a two-triangle plane, a flat gray reference, seed 12345, **one denoising step**, guidance 1, sparse retention 0.2, bf16, 2048-token chunks, and **synthetic zero text embeddings**. Native Qwen encoding was a separate check. These runs test the integration and memory path, not artistic quality or long-run throughput.

| Fixture | Observation |
| --- | --- |
| 512px, two albedo jobs before scoped weight reuse | 6.39 s / 4.77 s per single step; one manager request for both jobs |
| 512px, albedo + MR after scoped reuse | 4.35 s albedo / 4.52 s MR single steps; two task-group manager requests; 12 decoded views in 6.82 s |
| Native 2048px, before scoped reuse | 93.55 s single denoising step; 217.16 s total integration run |
| Native 2048px, after scoped reuse | 65.94 s single denoising step; six full 2048px views decoded in 82.77 s; 178.68 s total integration run |
| GGUF scoped weight reuse, real patched linear layer | Maximum output difference from native forward: 0.0 in the tested bf16 layer |
| 4096 atlas, before view/visibility pruning | 49.71 s base-color bake |
| Same 4096 fixture, after view/visibility pruning | 12.52 s base-color bake; identical observed fraction and passing color/orientation assertions |
| Real empty-prompt Qwen3-8B Q6_K | 32.71 s including loader/encoding |

The 2K fixture selected 15,138 target tokens, 15,138 normal tokens, 7,569 reference tokens and 512 text tokens, from 98,304 full target positions. Selected foreground occupied about 15.4% of the six-view canvas. A bulky mesh with more foreground can be substantially slower and may fail the capacity preflight.

The 2K sampling phase reported 6.26 GB peak allocated and **9.53 GB peak reserved** in PyTorch allocator accounting, despite an approximately 8.55 GB physical GPU. Windows virtual/shared GPU memory and allocator accounting mean this is **not** evidence that the whole job stayed inside dedicated VRAM. Native 2K is available but is not a fast preset on this machine. Device-wide peaks during render/decode, paging, thermal effects and transfer counts were not instrumented. Manager-request counts are not actual layer-transfer counts.

The before/after observations are single local runs, with possible differences in kernel-cache warmth, available RAM and laptop clocks. They are useful engineering evidence, not a controlled claim of a universal percentage speedup. The bake fixture has four empty camera views and one useful front-facing view, so it especially benefits from pruning; complex meshes need their own measurements.

## Not yet validated / not implemented

- No full artistic reference-to-texture quality run, matched numerical comparison against the original complete UltraTex pipeline, or controlled production step-count study.
- No browser import/visual layout check of the example graphs, and no end-to-end graph execution combining the retained Qwen loader cache with diffusion on 16 GB RAM. The core operations and their native node interfaces were tested separately.
- No native 4096 generation, six-million-face mesh benchmark, or 20-job real-model endurance test.
- No quality acceptance for seams, reference identity, back-side consistency, MR plausibility or thin/occluded surfaces.
- UV overlap validation is texel-based; coverage is occupied-atlas-texel coverage, not mesh surface area. No multi-material atlasing, UDIMs, arbitrary camera calibration, automatic rotated-rig reprojection, cross-view-set baking, surface-aware seam optimization or geometry-derived AO/normal baking.
- RAM settings bound the pack's small-array cache and enforce headroom checks; they do not impose a global cap on ComfyUI's retained model objects, IMAGE tensors or the OS file cache. Disk outputs/bundles are distinct from the spill budget.

The initial release is suitable for installation and controlled trials. Review these limits before treating it as a production replacement. The portable views/material-only workflows are useful independently of the generation path.

Raw reports are in `validation/`. Local absolute paths are shortened in the delivered copies; tensor counts, timings and test results are retained.
