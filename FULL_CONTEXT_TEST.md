> Historical release note. Current bake presets are Fast, Quality and Quality + edge repair; see README.md.

# UltraTex Breakthrough TEST — installation and rollback

This archive is installable, but the new full-context decoder is experimental and **disabled by default**. Read the separate engineering report for measured speed, memory, image differences and exclusions. It is not a certified 20–30% quality-preserving production improvement.

Measured combined TEST results: the first character-scene pair improved609.982→511.356s (16.2%) with comparable sampled memory. A deer-scene pair improved488.341→366.483s (25.0%) but increased dedicated peak by64MiB; that fails the strict memory-parity requirement. Full-context changes colors/context, and a mixed-workflow encoding-repeatability issue remains unresolved. Keep the mode disabled when strict quality/memory parity is required.

The final **FullContextTest preset uses native GGUF storage**, removing that earlier hybrid combination. One complete deer isolation run takes375.179s versus488.341s native (23.2%), with6692/399MiB dedicated/shared versus6724/399MiB native. Initial encoded inputs match. This is a single diagnostic run against the earlier control, with kernel-cache/order limitations; repeated performance/quality acceptance and this exact preset on the character scene remain unverified. The earlier combined results must not be presented as measurements of the final preset.

## Install

1. Stop ComfyUI. Preserve your current UltraTex custom-node folder outside `custom_nodes`, or rename it with ComfyUI's `.disabled` suffix. Preserve your models, workflows and settings.
2. Extract this ZIP so `ComfyUI/custom_nodes/comfyui_ultratex_modular/__init__.py` exists. Avoid a second nesting level and two active UltraTex copies.
3. Keep the existing PyTorch2.8.0+cu128/CUDA12.8 environment and model paths. This package does not require a Torch/CUDA upgrade or new profiler packages. Existing node dependencies still apply.
4. Start normally for the existing adaptive/tiled path. Models and runtime junctions are not included.

## Explicit modes

The optional PowerShell helper is `tools/Start-UltraTex.ps1`. Its default `Legacy` mode retains tiled decoding, disables architectural research flags and adds no startup allocator cap. `Control` uses original tiled architecture plus the explicit6144MiB native allocation cap used in the matched benchmarks. `Hybrid` is the historical GGUF storage research control, still tiled, with unqualified memory parity. `FullContextTest` uses native GGUF storage and adds full-context decoding.

Example from this package folder (supply your actual paths):

```powershell
./tools/Start-UltraTex.ps1 -Mode FullContextTest -ComfyRoot 'C:/path/ComfyUI' -PythonPath 'C:/path/python_embeded/python.exe'
```

Use `-DryRun` to inspect the command and environment without starting ComfyUI. The helper uses existing DynamicVRAM, two asynchronous offload streams and PyTorch attention. Add existing launch options with `-ExtraArgs @('--port','8188')`. It restores the invoking shell's environment after exit. It does not edit your launcher or install files.

For an existing launcher, the equivalent process environment before starting ComfyUI is:

```text
UTX_ARCH_RESEARCH=1
UTX_ARCH_GGUF_STORAGE=native
UTX_ARCH_POLYPHASE=0
UTX_FULL_CONTEXT_VAE=1
UTX_FULL_CONTEXT_STARTUP_CAP_MB=6144
```

Extension registration applies the allocation fraction after ComfyUI initializes Torch, before graphs only when explicitly requested. This fraction is not a hard bound on total driver reservations or other applications. Its default is inactive. The tested6144MiB startup cap is for this8GB test setup; do not infer validated settings or performance for other GPUs.

## Supported experimental scope and fallback

The full-context experiment is gated to tiled-requested2048 BaseColor/albedo jobs, foreground AutoencoderKL BF16,128×128 packed latents, CPU VAE output, CUDA/cudaMallocAsync, a valid explicit startup fraction and the tested physical-memory/headroom envelope. MR, other resolutions/interfaces, original performance mode, unsupported allocators or insufficient budgets retain the original decoder. BaseColor/MR, fast_detail/off_direct and MESH interfaces remain available through the existing path. MR has not been GPU-certified with a full-context decoder.

Eligible frames use full-image normalization and attention, bounded128-row convolution workspaces, reduced live residual buffers and native pool reclamation. The frame cap is at most5632MiB and respects a tighter existing cap. Native VAE OOM falls back to tiled output and disables further full-frame attempts for that decode operation. Scoped forwards, estimator, hooks and cap restore on normal completion/error. An explicit native cap remains process-wide until restart.

The sampled 8GB complete-workflow evidence does not guarantee safety under arbitrary external GPU pressure. Real6GB/16GB performance is unverified. A fallback run is not evidence of full-mode speed.

Full-context and tiled output use separate disk-cache identities. The model, LoRA, seed, sampling steps, camera fit, sparse retention and output resolution remain unchanged. Full-context output is not numerically equal to tiled output; downstream guided inputs can change as well. Retain TEST status until image/detail/seam checks on your scenes establish suitability.

## Rollback

Stop ComfyUI and restart with `-Mode Legacy` or remove the five environment variables above from your launcher. Restarting is required to restore the allocator's original startup state. Selecting runtime `performance_mode=original` disables scoped optimizations but cannot undo a process-wide startup cap without restarting.

For full source rollback, stop ComfyUI, move this package out of `custom_nodes`, restore your preserved previous UltraTex folder and start your original launcher. Previous delivery ZIPs are not overwritten. There is no model conversion or installation of new runtime binaries to reverse.
