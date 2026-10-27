# Third-party code and host dependencies

## UV-border and normal-preservation fixes (2.3.14rc1)

`core/uv_seams.py` is an independent NumPy/SciPy implementation of sparse
bilinear border constraints, with occupied texels fixed. No third-party seam
solver code is copied. Research references: Liu, Ferguson, Jacobson and Gingold,
Seamless (SIGGRAPH Asia 2017), https://cragl.cs.gmu.edu/seamless/ . The authors'
reference implementation is MIT, but is not bundled or called:
https://github.com/zfergus/seam-erasure/blob/master/LICENSE .

The mesh-import scene traversal, inverse-transpose normal transform, and export
normal-field consistency fixes are independently authored under GPL-3.0-only.
No new runtime dependency is introduced. GPL permits commercial use subject to
its terms; this does not replace any model or input-asset licence.

The separately supplied quality upscaler is the official RealESRGAN_x4plus.pth
v0.1.0 release, SHA-256
4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1.
Publisher: https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0 .
BSD-3-Clause notice is retained in vendor/licenses/REALESRGAN-BSD-3-Clause.txt.
The installable code ZIP contains no model weights; the delivery folder's
separate models/ directory contains this upscaler. No new diffusion weights.

## Material grain restoration (2.3.12rc1)

`core/material_grain.py` is an independently authored NumPy/SciPy implementation
of example-patch quilting, minimum-error overlap joins, chart-restricted colour
filtering and variance-normalized triplanar detail sampling. It retains this
project's GPL-3.0-only terms. No Embark or other third-party synthesis source,
binary, graph solver, checkpoint or network access is added. The example and
material mask are supplied by the user; their asset rights remain separate.
Commercial use of this code is permitted subject to GPL-3.0-only. Existing model
licences, especially the FLUX9B non-commercial model terms, are unchanged.

The rc4 CPU pool controller calls the separately installed `threadpoolctl`
3.6 API, under the BSD-3-Clause license. Commercial use is permitted under that
license; preserve upstream notices when redistributing the dependency.
[Upstream license](https://github.com/joblib/threadpoolctl/blob/master/LICENSE),
[thread-pool control API](https://github.com/joblib/threadpoolctl#setting-the-maximum-size-of-thread-pools).
No dependency source or model weights are bundled by this addition.

The 2.2.0rc2 `sharp_source` branch, row-streamed preparation and native-scale enhancement control are original implementations using the existing NumPy/SciPy/PyTorch APIs, under the project's existing GPL-3.0-only terms. No MVS-Texturing/OpenMVS code, graph solver, registration library or additional checkpoint is incorporated. No model weights are redistributed. The existing Real-ESRGAN BSD-3-Clause reference remains the tested selectable upscaler; the code does not automatically substitute weights.

Project-authored UltraTex Modular v2 code is offered under GNU GPL-3.0-only, subject to copyright ownership. Previously published MIT-licensed editions retain their prior grants. Vendored third-party MIT/Apache-2.0 files retain their upstream copyright/license notices; their texts are reproduced alongside the code. No model weights are redistributed.

## Vendored sources

- `vendor/optim.py` comes from [visualbruno/ComfyUI-UltraTex](https://github.com/visualbruno/ComfyUI-UltraTex/tree/8137da52ed8352305cb7825b39c36a250bf2507a), commit `8137da52ed8352305cb7825b39c36a250bf2507a`, path `ultratex_core/optim.py`. MIT notice: `vendor/UPSTREAM_LICENSE`. Local modification: scoped `linear_chunks` reuses the installed GGUF loader's dequantized, adapter-patched weights within each single-block linear chunk loop; original execution remains the fallback for other layers/hooks. Weights are released at scope exit, including exceptions.
- `vendor/sparse_attention/` is the same snapshot's vendored UltraTex/SLA attention implementation. Apache-2.0 notice: `vendor/sparse_attention/LICENSE.txt`; upstream modifications: `vendor/sparse_attention/PATCHES.md`. These files are otherwise unmodified.
- Further local changes in `vendor/optim.py` (0.1.0.post1 / 0.1.1): require complete chunk-patch coverage instead of silent fallback; allocate Q/K/V independently; reuse scoped linear weights in supported double-block MLPs; preallocate those MLP outputs. Unsupported layouts and hooks retain their original chunked execution. Native-block parity and real-model comparisons are recorded in the validation reports.
- Camera matrices/conventions, foreground-mask dilation, six-tile packing, FLUX.2 position IDs, and UNO timestep equations were adapted from that MIT-licensed wrapper and the [UltraTex reference implementation](https://github.com/yiboz2001/UltraTex/tree/a726678f3f3631fe2c5e3d1fb4597114784c7a8b). The original MIT notice is preserved in `vendor/UPSTREAM_LICENSE`.
- In 0.1.2, internal depth-edge feathering, narrow white-rim rejection and four-neighbour 3D hole filling in `core/bake.py` are adapted from visualbruno's MIT-licensed `ultratex_core/bake.py`. The implementation retains tiled projection and disk-backed evidence. Routine chunking logging was removed from `vendor/optim.py`; numerical sampler optimizations are unchanged.

## Host libraries used through their APIs

ComfyUI and ComfyUI-GGUF remain separately installed host components. This pack calls their native model manager, patcher, text/VAE operations, MESH/FILE_3D APIs, and GLB exporter. Their source is not copied into this package. The host's licenses continue to apply to its own distribution.

The added numerical/graphics dependencies are NumPy (BSD), SciPy (BSD), Pillow (HPND), einops (MIT), ModernGL (MIT), glcontext (MIT), and psutil (BSD). Triton/SLA use their upstream licenses. See installed distributions for full notices and `component_manifest.json` for tested versions. These code licenses do not replace the separate terms of model weights.

## Model terms

See `component_manifest.json` and upstream model cards. The selected FLUX.2 Klein base 9B model uses BFL's FLUX non-commercial model license; model-use permissions and generated-output permissions are separate. This package does not grant commercial model-use rights, add a watermark, claim generated-output ownership, or introduce a hosted service. Quantized-file origin and separate adapter/decoder artifact terms must be checked against the provider of the files you use.

## Visibility Projection v3 (v0.3.0)

`core/visibility_v3.py`, the camera face-ID rendering/cache and its integration into the baker were independently authored for this project using NumPy and existing ModernGL APIs. No OpenMVS/AGPL source or dependencies are incorporated. Older experimental `core/smart_fusion.py` and `core/edge_fix.py` modules are **not shipped** in this version.

## Fast Surface Texture Baker (v0.4.0)

`core/fastbake.py`, `fast_baker.py` and the v0.4 tests/tools were written independently for this project, now distributed with the project under GPL-3.0-only; original development notes described them as MIT. They use only NumPy (BSD), SciPy (BSD), Pillow (HPND) and PyTorch (BSD-style), all already required by ComfyUI or this pack. No rasteriser library (no nvdiffrast), no AGPL/GPL code and no new model weights. The software UV rasteriser, plane-consistent visibility test, two-band fusion and surface completion are original implementations of standard techniques. They are not copies of another project's code.

Test assets are generated procedurally inside the tests. No third-party meshes or textures are shipped.

## View 0 anchor / coherent bake (2.2.0rc3 TEST)

The new source-anchor, geometry-filter, patch-graph, metric alpha-expansion and
bounded log-Poisson code was implemented independently under the project's
GPL-3.0-only license. Commercial use is permitted subject to that license.
No MVS-Texturing or third-party optimizer source/binary or model weights are
included. Runtime operations use existing PyTorch, NumPy and SciPy dependencies;
no package or environment upgrade is introduced. SciPy publishes BSD-3-Clause
terms: https://github.com/scipy/scipy/blob/main/LICENSE.txt . The MVS-Texturing
BSD-3-Clause source was researched as a reference, not integrated:
https://github.com/nmoehrle/mvs-texturing/blob/master/LICENSE.txt .

## Fast Detail bake (v0.4.1)

`upscale_mode=fast_detail`, the bounding-box VOSR2 call of `views_first`, the new tests and `tools/detail_benchmark/` were written independently for this project under the project GNU GPL-3.0-only license. They implement published, standard techniques from scratch (Keys cubic convolution, a min/max anti-ringing clamp and band-split blending in the spirit of Burt & Adelson); no third-party source is copied.

No new runtime dependency and no model weights: the baker only uses PyTorch operations (`grid_sample`, `pad`, `max_pool2d`, arithmetic) that ComfyUI already requires. QA-only tools that the node never imports: `xatlas` (MIT) unwraps the benchmark mesh; `spandrel` (MIT) and Real-ESRGAN weights (BSD-3-Clause) were used once to evaluate AI upscalers and are neither shipped nor required. The benchmark's ground-truth texture is generated procedurally (DejaVu fonts from the host system if present); no third-party textures or meshes are shipped.


## v2.3 Q4 expansion candidate
The new UltraTex Q4_0 expansion kernel uses the already-installed Triton runtime
(MIT license). It introduces no additional dependency, copied third-party kernel,
model artifact, or noncommercial dependency. Existing ComfyUI-GGUF Apache-2.0,
SLA, UltraTex and model/LoRA terms remain unchanged.


## Optional view-enhancement dependencies (checked2026-10-20)

The latent-tile scheduling adapter in core/super_resolution.py follows the VOSR / ComfyUI-VOSR2 reference implementation, adapted for native UTX bundles, exact targets and foreground dependency scheduling. The external backend remains separately installed. Its Apache-2.0 text is retained in vendor/licenses/VOSR2-APACHE-2.0.txt. No checkpoint or third-party architecture source is bundled here.

- VOSR project code and project-authored checkpoints: Apache-2.0; bundled third-party components retain their own terms. [VOSR model card](https://huggingface.co/CSWRY/VOSR/blob/main/README.md), [code license](https://github.com/cswry/VOSR/blob/main/LICENSE).
- VOSR2's converted Qwen Image VAE follows its Wan2.1/Qwen upstream terms; those projects publish Apache-2.0. [Wan license](https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt), [Qwen model card](https://huggingface.co/Qwen/Qwen-Image).
- DINOv2 code/weights: Apache-2.0 per [upstream repository](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).
- Official BiRefNet code and official ZhengPeng7 model card: MIT. [code license](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE), [model card](https://huggingface.co/ZhengPeng7/BiRefNet). This is not an authorization for every BiRefNet-derived model; BRIA RMBG checkpoints have separate terms. The local generic filenames supplied here do not authenticate checkpoint provenance.
- Optional Real-ESRGAN research comparison: publisher repository is BSD-3-Clause; checkpoint taken from its official release. [license](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE), [release](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0). Research weight is not in the installable ZIP. Other UPSCALE_MODEL checkpoints require their own terms review.

These observations do not confer unconditional commercial rights to a complete pipeline or its outputs. Existing FLUX9B / UltraTex adapter obligations in the original notices still apply, as do rights in the input assets. No change to those models or terms is implied by the new node.
