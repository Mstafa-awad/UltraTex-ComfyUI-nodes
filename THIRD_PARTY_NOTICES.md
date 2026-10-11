# Third-party code and host dependencies

The node pack's new code is MIT licensed. No model weights are redistributed.

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
