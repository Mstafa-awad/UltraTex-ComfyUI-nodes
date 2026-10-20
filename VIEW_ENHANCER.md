# UTX 2.5 â€” Intelligent View Enhancer (TEST)

Install this entire `comfyui_ultratex_modular` folder in ComfyUI/custom_nodes, replacing the previous folder after making a backup outside custom_nodes. Restart ComfyUI. No Torch/CUDA/driver/model upgrade is needed. Keep the supplied original ZIP for exact rollback. This package includes no checkpoints.

## Connections

For the 2.2.0rc2 Native model scale target, `sharp_source` bake and true optional 8K atlas, see [SHARP_SOURCE.md](SHARP_SOURCE.md). The retained options below remain available.

1. UTX2 `views` â†’ UTX2.5 `views`.
2. UTX2 `generation_setup_to_bake` â†’ both UTX2.5 `setup` and UTX3 `albedo_setup`.
3. UTX2.5 `enhanced_views` â†’ UTX3 `albedo_views`.
4. Keep the MR branch connected directly to UTX3. Keep UTX4 Cache Cleaner after all consumers of the disk-backed views, including Save Views.

`enabled=false` is the default and restores the original bundle. The separate `original_views` output also provides exact originals. Camera fitting/FOV, generation settings, depth, normals, UVs and the generator are unchanged. Enable `debug_preview` only when a 512px image/mask batch is useful; it is not the baking data.

## Start without AI

Enable the node; choose geometry masks, crop on, square crop off, None, Original. The crop removes unused background without resizing source pixels. Bounds/padding align to the existing mip grid. The cropped images can look larger in previews but contain no newly recovered detail. UTX3 supports the transforms in fast_detail, off_direct and the new optional detail_consistent mode.

Keep fast_detail for the existing reconstruction. detail_consistent uses world-to-UV differential area for filter footprints, a full mip pyramid, and a narrower continuous handover between views for mid-frequency detail; it may reduce double patterns, but can expose disagreements and is not a universal quality upgrade. Both original modes remain available. Raising UTX3 texture_size to4096 is a separate existing option: compare the memory/time and image quality independently of this node.

## VOSR 2.0

Install ComfyUI-VOSR2 separately and supply its existing complete local VOSR2 bundle. This node never downloads weights. Selecting VOSR2 loads once across all active views, then releases its automatically owned model patchers through ComfyUI to avoid retaining multi-GB weights for the next job. Connected native loader models remain loader-owned. Alternatively connect its native VOSR2_MODEL loader output. Default automatic loading uses BF16 DiT/DINO with the backend's original FP32 Qwen VAE. Native 512px DiT tiles and overlap are retained. Target4096 is optional; Original does not enlarge the crop.

VOSR resizes the native crop directly to its exact final dimensions before restoration, so fractional factors do not require a larger integer-scale intermediate. Rectangular crops are enhanced before letterboxing, avoiding AI processing of the letterbox. Pixels outside the dependency closure of relevant VAE decode tiles may skip DiT inference; VAE attention and the color-alignment guard are included. The RGB/mask canvas retains an exact reversible affine transform for the baker.

Serial processing is deliberate: a measured BF16 batched-tile experiment changed pixels significantly and was rejected. More VRAM does not enable that unvalidated change. Available memory controls Comfy placement; Conservative retains serial operation. A CUDA OOM retry uses256px VAE tiles and is explicitly reported because tile context can change. Very large targets are rejected before unsafe allocations. Physical6GB/16GB devices have not been certified.

AI can invent iris/fur/armor patterns. Strength blends the estimate with the source; zero skips AI. Never use this node on MR, normals, depth or scalar material data. Compare matching renders and original_views before adopting an AI result.

## Other optional models

Comfy upscale model accepts the native UPSCALE_MODEL from Load Upscale Model. It uses the checkpoint's native scale and one final accurate resize. No model is silently substituted for VOSR2. Model-specific license/provenance remains your responsibility; see THIRD_PARTY_NOTICES.

Geometry masks avoid segmentation cost. For optional BiRefNet, select BiRefNet mask or hybrid mask and connect BACKGROUND_REMOVAL from ComfyUI's Load Background Removal Model, or an ordered foreground MASK batch (foreground=1) from its Remove Background node. Supplied masks take precedence. Geometry protects silhouette bounds even if segmentation fails; the segmentation mask does not erase camera-visible geometry. Both segmentation modes use that safety union. No inference runs in geometry mode.

## Saved bundles and rollback

Save Views preserves enhanced RGB, masks, transforms and originals in a version2 portable bundle. Load with this version of UltraTex. Original bundles stay version1-compatible. Legacy advanced baking functions reject enhanced bundles rather than silently sampling incorrectly. Restore originals before using a baker that does not understand the metadata.

Disable UTX2.5 or connect original_views to return to the existing pipeline. To roll back the entire release, close ComfyUI and replace this folder with the separate original ZIP. Keep backups outside custom_nodes so both versions do not register duplicate nodes.


## Measured fast alternative

RealESR-general-x4v3 (native4x SRVGG, official Real-ESRGAN release) is a selectable **Comfy upscale model** example. A clean GPU test enhanced12 wolf views to4096 longest-side crops in25.5s (first ten21.9s), versus373.7s for12 VOSR2 views. These are enhancement-stage times, not total FLUX+bake times. AI4K images also cost more to bake. Natural-image recovery is not ground-truth detail; inspect iris/eyelid changes. The optional workflow expects realesr-general-x4v3.pth in the native upscale_models folder. The researched weight remains outside the installable package; its official download and terms are in THIRD_PARTY_NOTICES. No model is installed automatically.
