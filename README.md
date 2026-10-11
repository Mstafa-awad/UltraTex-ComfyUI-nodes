# UltraTex Modular 0.2.0

**New in v0.2:** [UPGRADE_0.2.md](UPGRADE_0.2.md) contains the installation, upgraded triangle-artifact/UV-mapping code changes, and four importable **`workflows/`** JSONs. Start with `workflows/01_NATIVE_2048_6V_8GB.json` for 2048-native generation or `workflows/03_FAST_1024V_TO_2048_ATLAS.json` for a cheaper baseline. The `02_NATIVE_2048_12V_GUIDED.json` workflow adds a reprojected, guided 45° pass instead of independent generated views. **Native 2048 on an 8GB GPU has not been performance-tested here and may offload or run out of memory.**

*The remaining README retains background and measurements from the v0.1.2 baseline; historical validation logs do not validate v0.2.*

Three main ComfyUI stages for FLUX.2 Klein **base 9B** + UltraTex: **Prepare Texture → Generate Views → Bake Texture**. Keep your UV node, choose 6 or 12 views, preview every generated image, and save a textured GLB. The original 27 nodes remain available for advanced workflows. GGUF is tested locally; compatible native ComfyUI MODEL loaders are also accepted, but FP8 has not been benchmarked here.

This revision fixes normals at UV splits that caused the reported blocky face, improves projection and surface fill, removes routine debug output, and fixes the false RAM-headroom rejection. Start with **00_simple_6_views.json** (1024px generated views / 2048px texture). **00_simple_12_views.json** adds a 45° second pass. See [FIXES_0.1.2.md](FIXES_0.1.2.md) for the same-view character comparison and limitations. Native 2048 generation remains a much heavier workload.

## Install later

1. Copy the entire `comfyui_ultratex_modular` folder into `ComfyUI/custom_nodes/`. Keep the folder name and its `__init__.py` at that level.
2. Use a recent ComfyUI build with the V3 API, native `MESH`, `FILE_3D`, `Load3DAdvanced`, and `SaveGLB`. The tested revisions are recorded in [component_manifest.json](component_manifest.json).
3. Install [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) if using a GGUF diffusion model or text encoder.
4. Install this package's requirements **with ComfyUI's Python**, then restart ComfyUI.

For a Windows portable installation, run this from the folder containing `python_embeded` and `ComfyUI`:

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\comfyui_ultratex_modular\requirements.txt
```

The requirements do not replace PyTorch or download models. The original ComfyUI-UltraTex node pack is not required; this pack has distinct `UTX` node IDs and can coexist with it.

Sparse generation requires Triton and an NVIDIA CUDA GPU. The tested local runtime already had a working `triton-windows`; retain a working installation. For a fresh setup, follow the [Triton Windows compatibility guide](https://github.com/woct0rdho/triton-windows#installation) and pair it with your PyTorch minor version. Its documented pairing for PyTorch 2.8 is Triton 3.4. The test machine instead had Triton 3.8.0.post28; its successful test is recorded, without asserting that this pairing is generally supported. Triton is deliberately excluded from automatic requirements. A missing/broken sparse backend raises an explicit error; `dense_debug` is an opt-in diagnostic mode and may be much slower or use too much memory.

ModernGL needs an OpenGL 3.3 context. Baking and material-only graphs do not load diffusion weights. Material-only graphs do not need a raster context.

## Existing model files

Use stock loaders / ComfyUI-GGUF loaders with these files or their verified equivalents:

| Component | ComfyUI location | Example selection |
| --- | --- | --- |
| **Base**, undistilled Klein 9B GGUF Q4 | `models/diffusion_models/` | `flux-2-klein-base-9b-Q4_0.gguf` |
| Qwen3-8B encoder | `models/text_encoders/` | `Qwen3-8B-Q6_K.gguf`, CLIP type **flux2** |
| FLUX.2 VAE | `models/vae/` | `flux2-vae.safetensors` |
| UltraTex albedo adapter | `models/loras/` | `ultratex_flux2.safetensors` |
| UltraTex MR adapter | `models/loras/` | `ultratex_flux2_mr.safetensors` |
| FLUX.2 foreground decoder | `models/ultratex/` | `ultratex_flux2_decoder.pt` |

Sources: [BFL base model](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), [UltraTex artifacts](https://www.modelscope.ai/models/Yibo-Zhang/UltraTex), [upstream wrapper's model table](https://github.com/visualbruno/ComfyUI-UltraTex#models). The adapter source paths are `flux2/lora/dit_lora.safetensors` and `flux2_mr/lora/dit_lora.safetensors`; rename them distinctly. Use `flux2/decoder.pt` for the foreground decoder. Do not select FLUX.1, a Klein 4B model, or the distilled 9B model.

Architecture and adapter-pair coverage are checked at runtime. Known albedo/MR file hashes and explicit filename labels are checked against the task before execution; this also catches renamed copies of the tested adapters. Unknown adapter provenance is reported as user-selected. Matching shapes alone cannot certify that a checkpoint is undistilled. The manifest records hashes of the files tested locally; weights are not included in this package.

The adapter filenames were initially swapped on the test installation; the user has now corrected them, and both hashes were reverified. Final examples use `ultratex_flux2.safetensors` for albedo (published SHA-256 `3b191517...`) and `ultratex_flux2_mr.safetensors` for MR (`13e94f5a...`). The node trusts published hashes over names. This package does not rename model files. See [FIXES_0.1.1.md](FIXES_0.1.1.md) for the investigation and publisher verification.

## Start with an example

Drag a JSON from `examples/` into ComfyUI. Choose your GLB and reference image, and check the loader selections before queuing. Filenames in examples are placeholders. `examples/api/` contains corresponding API templates for scripted use.

| Workflow | Purpose |
| --- | --- |
| `00_simple_6_views.json` | Recommended: three main stages, all-view previews, 1024px views / 2048px texture |
| `00_simple_12_views.json` | Two six-view passes, second rig at 45°; roughly twice the generation work |
| `01_preview_512.json` | Five-step 512px preview; 1024px atlas |
| `02_native_2k_4k_atlas.json` | Six native 2048px views; separate 4096px atlas; 25 steps as an uncalibrated starting point |
| `03_albedo_and_mr.json` | Shared base GGUF with separate albedo/MR branches, one combined workset |
| `04_bake_saved_views.json` | Rebake a saved view bundle without generation models |
| `05_materials_only.json` | Apply existing base color/roughness/metallic IMAGEs; other maps are optional sockets |
| `06_twenty_variations.json` | Sample 20 jobs before decoding; save all 120 views, preview/bake a selected job |
| `07_native_mesh_integration.json` | Replace the GLB adapter connections with your upstream native MESH |
| `08_external_views_bake_only.json` | Bake six externally supplied, calibrated images without generation models |
| `09_1k_views_2k_atlas.json` | Advanced equivalent: six 1024px views, 2048px atlas, 25 steps, surface fill and gutters |
| `10_native_fp8_1k_views_2k_atlas.json` | Optional native diffusion-loader template; requires separately installed base 9B FP8 weights; unbenchmarked locally |

The examples' required connections and socket types were validated against installed schemas. Their browser layout has not been inspected. All examples preview every view and use surface fill without a Diagnostics node. The simple graph returns both the texture and the completed GLB from Bake Texture; advanced graphs keep map metadata connected to material assembly.

Use a single static mesh with one UV set in `[0,1]` and unique, non-overlapping islands. Connect the original MESH to both **Prepare Geometry** and **Apply Materials**. Preparation normalizes a separate conditioning copy; exported geometry retains its input size, orientation, topology and UVs. If UVs are missing or unsuitable, explicitly run ComfyUI's **Unwrap Mesh UVs** first. No unwrap/remesh happens silently. GLB import rejects multiple materials, animation, skinning, morph targets and extra UV sets before invoking the native importer. Native MESH producers are responsible for retaining their source material assignments.

For references, a transparent front view works best. Connect both outputs of **Load Image** to **Prepare Reference** and keep `mask_polarity=background` (ComfyUI's mask convention). A foreground segmentation mask instead uses `foreground`, where 1 means object. An opaque reference without a mask treats the whole image as foreground. Empty text conditioning is the image-guided default. Leave `negative` disconnected unless needed; no CFG second pass occurs without a distinct negative. `guidance` only affects that second pass.

## Modular graph and data conventions

```text
native MESH -> Prepare Geometry -> Plan Views -> Render G-buffers
reference IMAGE/MASK ----------------------------> Prepare Reference
VAE + reference + G-buffers -> Encode Conditions -> Build Foreground Tokens
CLIP -> Text Encode ----------------------------> Build Workset
native/GGUF MODEL -> Apply Texture LoRA ----------> Sample Latents
VAE -> Apply Foreground Decoder + sampled latents -> Decode Views
                     +-> Inspect / Split Views -> standard IMAGE/MASK
                     +-> Save View Bundle -> reload later
MESH context + G-buffers -> Prepare Atlas Projection
atlas + decoded/external views -> Bake Map -> optional Fill / Repair Atlas
original MESH + independent IMAGE maps -> Apply Materials -> Create 3D File -> Save 3D
```

- View order is **front, left, back, right, top, bottom**. Internal joint packing is `[[front,left,right],[back,top,bottom]]` according to the canonical rig. Always use Inspect / Split Views for ordered IMAGE outputs instead of reshaping latent storage yourself.
- `render_size`, `sample_size`, and `atlas_size` are independent. Baking a 4096 atlas does not mean 4096 generation. Foreground token counts and actual latent/image dimensions are reported separately.
- `IMAGE` uses ComfyUI BHWC float `[0,1]`; `MASK` uses BHW. Decoded collections store RGB uint8 arrays on disk. Inspect only the views you need to bound materialized memory.
- Native ComfyUI/glTF UVs use `v_down`. If your producer explicitly uses `v_up`, choose it consistently in geometry and materials. Apply Materials then converts that output branch's UV representation to native `v_down`; the input mesh is unchanged. No UV flip is hidden in an IMAGE socket.
- Base color and emissive are interpreted as sRGB and blended in linear light. Roughness, metallic and AO are linear data. UltraTex MR has roughness in **G** and metallic in **B**. Bake those semantics from an `mr` view set or use Unpack MR. Export packs glTF ORM as R=AO, G=roughness, B=metallic.
- Tangent normals default to OpenGL. Selecting DirectX flips green before normalization. Normal maps need the mesh/renderer tangent basis to agree.
- Every material map is an independent optional IMAGE input. Missing channels inherit existing textures/factors. Trellis2/Pixal3D/external maps must already use this mesh's UV layout. Optional `*_info` sockets validate an UltraTex map's UV identity; absent metadata cannot establish provenance.
- For opacity, supply a scalar IMAGE and select `MASK` or `BLEND`. Use **UTX Create 3D File** before stock Save/Preview 3D to preserve alpha; the ordinary native mesh exporter does not understand the pack's opacity metadata.

## Fast execution and bounded storage

The sampler retains foreground tokens only, uses the trained sparse-attention path, and processes CFG sequentially when needed. GGUF weights remain quantized in storage. Single-stream linear layers and supported double-block MLPs reuse dequantized, adapter-patched weights across token chunks, then release them. Separate Q/K/V storage avoids retaining the combined buffer through V. Token chunking bounds scratch space; full-sequence residual/attention/MLP buffers still exist, so chunking cannot make every resolution fit.

**Build Workset** creates jobs with seeds `seed+i`. Append another prepared workset through `previous`. **Sample Latents** finishes the whole workset before any decode. It groups compatible albedo/MR jobs and requests the model manager once per task group, with one bounded OOM retry. Internal offloading/transfers are controlled by ComfyUI; a single request does not promise full model residency or zero transfers. For albedo+MR, both adapter branches must originate from the same backbone loader. Decoding is serial, one view at a time.

**Configure Runtime** controls:

| Setting | Behavior |
| --- | --- |
| RAM cache | Aggregate small-array LRU; default 256 MiB. This does not cap model storage, native IMAGE outputs or OS file caching. |
| Disk spill | Default `output/.ultratex_cache`, with a 16 GiB array budget. Complete arrays are committed atomically; live arrays are never silently evicted. |
| Headroom | Host RAM setting is a cache-trimming target; preflight retries after collection and uses a bounded allocation margin. GPU setting remains estimated activation headroom. |
| Token chunk | Smaller reduces scratch allocations but increases launch overhead. Examples use 4096, tested on the actual character at 1024px. Native 2K may need smaller chunks. |
| Atlas tile | Default 512. Raster geometry is cached and reused across semantic maps and rebakes. |
| Decode tile | Default 64 latent pixels; tiled VAE decode favors fitting in memory. Larger tiles can trade memory for speed. |

The normal/reference encodings, view G-buffers, UV raster tiles and baking evidence are cached. Completed job latents survive interruption and can be reused while model/encoder identities remain valid in the current process. Re-loading/replacing models invalidates that identity; cross-restart latent resume is not offered. The workset progress manifest identifies completed jobs and failures. Portable **Save View Bundle** exports JSON + NPY (no pickle) with images, masks, depths and cameras; **Load View Bundle** works across restarts. Exported bundles are separate outputs and are not covered by the spill budget.

Use a local SSD for spill; avoid an actively synchronized folder for large intermediates if possible. Clear `.ultratex_cache` only after no graph is using those handles. A full disk budget produces an actionable error instead of deleting live data. Sparse backend failures never silently change attention semantics. Cancellation is checked between jobs, diffusion steps, views and bake tiles; an active GPU kernel cannot be interrupted midway.

## Baking and repair

Baking combines depth visibility, incidence angle, silhouette and internal occlusion edges. Continuous sharpened weights prefer well-facing views without an abrupt best-view switch. Per-view linear/scalar evidence is prepared once, then sampled from disk-backed arrays across UV tiles. Depth tolerance is measured in normalized mesh space (bounding radius 0.5), not source metres. Coverage is the fraction of occupied **UV texels**, not mesh surface area, and confidence represents geometric view quality, not model uncertainty.

**Bake Map** extends valid texels into an 8px gutter by default, preventing black filtering borders. This changes only texels outside occupied islands. `edge_pixels` changes projection weighting near silhouettes and internal depth edges; it does not pad UV gutters. **Bake Texture** combines projection, surface fill, padding and export, and can bake both calibrated rigs together.

An `existing_map` fills unobserved texels. Advanced **Fill / Repair Atlas** offers `none`, `nearest_surface` and legacy `nearest_in_chart`. Examples use `nearest_surface`: four nearby observed texels in 3D estimate hidden colors while preserving observed detail, following the reference baker. Disable filling to inspect missing coverage. Filling cannot recover unseen details. Closely packed islands can still bleed at lower mip levels, so UV spacing matters. Overlap checks detect collisions at atlas texel centres, not subpixel overlaps. Arbitrary uncalibrated photos, UDIMs, overlapping/mirrored islands and learned hole completion remain unsupported.

## Reproduce checks

From the directory containing this package, using an appropriate Python:

```powershell
python -m unittest comfyui_ultratex_modular.tests.test_core -v
```

With ComfyUI's Python (replace paths):

```powershell
& 'C:\ComfyUI_portable\python_embeded\python.exe' .\comfyui_ultratex_modular\tools\smoke_comfy.py --comfy-root 'C:\ComfyUI_portable\ComfyUI' --sparse --atlas-size 4096 --report smoke.json
& 'C:\ComfyUI_portable\python_embeded\python.exe' .\comfyui_ultratex_modular\tools\check_models.py --comfy-root 'C:\ComfyUI_portable\ComfyUI' --sample --size 512 --jobs 2 --report models.json
```

The second command reads the exact example model filenames and uses synthetic zero text embeddings for a one-step integration test. It does **not** assess generated-texture quality or load the Qwen encoder. It skips sampling if less than 7 GiB host RAM is available and records that explicitly. `--size 2048 --jobs 1` exercises native 2K, which can take several minutes. No test installs the node pack into ComfyUI or downloads weights.

## License and remaining scope

New code is MIT; vendored kernels retain Apache-2.0 and their notices. The node pack adds no hosted dependency, watermark, or output-ownership term. Model terms are separate: FLUX.2 Klein base 9B is under BFL's non-commercial model license. See the linked source license and [component_manifest.json](component_manifest.json); this package does not grant commercial model-use permission.

This version supports 6/12-view generation and joint baking of both rigs. It does not include arbitrary calibrated-camera import, automatic reprojection/refinement of the second pass, surface-area coverage, multi-material atlasing, geometric AO/normal-map generation, or a quality preset validated across many assets. Advanced six-view init/preservation tensors require already aligned inputs. Native 4096 generation remains experimental and untested. Synthetic smoke checks are not artistic-quality benchmarks.
