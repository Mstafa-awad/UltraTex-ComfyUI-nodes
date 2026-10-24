from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VERSION = 1


@dataclass(frozen=True)
class RuntimePolicy:
    spill_directory: str
    ram_cache_mb: int = 256
    disk_budget_gb: float = 16.0
    system_headroom_mb: int = 2048
    gpu_headroom_mb: int = 768
    token_chunk: int = 4096
    activation_factor: float = 12.0
    atlas_tile: int = 512
    decode_tile: int = 64
    performance_mode: str = 'adaptive'
    max_vram_mb: int = 0
    adaptive_diagnostics: bool = False


@dataclass(frozen=True)
class ArrayRef:
    path: str
    shape: tuple[int, ...]
    dtype: str
    digest: str

    def read(self):
        import numpy as np
        array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != self.shape or array.dtype.str != self.dtype:
            raise ValueError(f"UltraTex: corrupt cache array {self.path}")
        return array


@dataclass(frozen=True)
class MeshContext:
    mesh: Any = field(repr=False, compare=False)
    vertices: Any = field(repr=False, compare=False)
    faces: Any = field(repr=False, compare=False)
    normals: Any = field(repr=False, compare=False)
    uvs: Any = field(default=None, repr=False, compare=False)
    geometry_id: str = ""
    atlas_id: str = ""
    centre: tuple = (0., 0., 0.)
    scale: float = 1.
    uv_convention: str = "v_down"

    def _comfy_cache_tensors(self):
        return self.mesh._comfy_cache_tensors()


@dataclass(frozen=True)
class ViewPlan:
    geometry_id: str
    render_size: int
    sample_size: int
    matrices: tuple
    identity: str
    rotation: tuple
    names: tuple = ("front", "left", "back", "right", "top", "bottom")
    active_indices: tuple = (0, 1, 2, 3, 4, 5)  # fixed six-slot DiT grid; 10-view pass activates only side slots


@dataclass(frozen=True)
class GBuffer:
    plan: ViewPlan
    normals: tuple[ArrayRef, ...]
    depths: tuple[ArrayRef, ...]
    masks: tuple[ArrayRef, ...]


@dataclass(frozen=True)
class Reference:
    rgba: ArrayRef
    identity: str
    transform: dict


@dataclass(frozen=True)
class TextureSetup:
    context: MeshContext
    gbuffers: tuple[GBuffer, ...]
    reference: Reference
    runtime: RuntimePolicy
    # Optional pictures for other views of the first (0-degree) rig: ((view_index, Reference), ...), view_index 1..5.
    extra_references: tuple = ()
    view_reference_strength: float = 0.5   # share of the first-pass sampling steps those views are held to their picture

    def _comfy_cache_tensors(self):
        return self.context._comfy_cache_tensors()


@dataclass(frozen=True)
class ModelInfo:
    base_identity: str
    task: str
    adapter_hash: str
    strength: float
    patch_size: int = 1
    axes: int = 4
    channels: int = 128
    hidden: int = 4096
    latent_format: str = "Flux2"


@dataclass(frozen=True)
class EncodedConditions:
    gbuffer: GBuffer
    normal_tiles: tuple[ArrayRef, ...]
    reference: ArrayRef
    background: ArrayRef
    reference_mask: ArrayRef
    masks: tuple[ArrayRef, ...]
    init_tiles: tuple[ArrayRef, ...] = ()
    keep_masks: tuple[ArrayRef, ...] = ()
    encoder_id: str = ""


@dataclass(frozen=True)
class TokenPack:
    conditions: EncodedConditions
    foreground_indices: ArrayRef
    references: ArrayRef
    position_ids: ArrayRef
    normal_count: int
    reference_count: int
    grid_h: int
    grid_w: int
    init_tokens: ArrayRef | None = None
    keep_weights: ArrayRef | None = None
    keep_steps: int = -1      # hold keep_weights for this many sampling steps; -1 = every step (guided 45-degree pass)

    @property
    def target_count(self):
        return self.foreground_indices.shape[0]


@dataclass(frozen=True)
class Job:
    identity: str
    pack: TokenPack
    seed: int
    positive: ArrayRef
    negative: ArrayRef | None
    task: str


@dataclass(frozen=True)
class Workset:
    jobs: tuple[Job, ...]


@dataclass(frozen=True)
class SampleResult:
    job: Job
    foreground: ArrayRef
    report: dict


@dataclass(frozen=True)
class Samples:
    results: tuple[SampleResult, ...]
    manifest_path: str


@dataclass(frozen=True)
class ImageTransform:
    """Pixel-edge affine map; integer pixel indices refer to their centres (+0.5).

    Camera/depth remain in original coordinates. The enhanced canvas is square;
    rectangular crops are aspect-preserved with explicit letterbox padding.
    """
    original_width: int
    original_height: int
    left: int
    top: int
    width: int
    height: int
    output_size: int
    resized_width: int
    resized_height: int
    pad_left: int = 0
    pad_top: int = 0

    def __post_init__(self):
        vals = (self.original_width, self.original_height, self.width, self.height,
                self.output_size, self.resized_width, self.resized_height)
        if any(not isinstance(v, int) or v <= 0 for v in vals):
            raise ValueError("Invalid positive image transform dimensions")
        if min(self.left, self.top, self.pad_left, self.pad_top) < 0:
            raise ValueError("Negative crop/padding")
        if self.left+self.width > self.original_width or self.top+self.height > self.original_height:
            raise ValueError("Crop exceeds original image")
        if self.pad_left+self.resized_width > self.output_size or self.pad_top+self.resized_height > self.output_size:
            raise ValueError("Resized crop exceeds enhanced canvas")

    def uv(self, u, v):
        return (((u*self.original_width-self.left)*self.resized_width/self.width+self.pad_left)/self.output_size,
                ((v*self.original_height-self.top)*self.resized_height/self.height+self.pad_top)/self.output_size)

    def pixels(self, x, y):
        """Original pixel-centre index -> enhanced pixel-centre index."""
        u, v = self.uv((x+.5)/self.original_width, (y+.5)/self.original_height)
        return u*self.output_size-.5, v*self.output_size-.5

    def inverse_pixels(self, x, y):
        return (((x+.5-self.pad_left)*self.width/self.resized_width+self.left-.5),
                ((y+.5-self.pad_top)*self.height/self.resized_height+self.top-.5))


@dataclass(frozen=True)
class ViewSet:
    plan: ViewPlan
    images: tuple[ArrayRef, ...]
    masks: tuple[ArrayRef, ...]
    task: str
    job_id: str
    gbuffer: GBuffer
    sampling: tuple[ImageTransform, ...] = ()
    original_images: tuple[ArrayRef, ...] = ()
    original_masks: tuple[ArrayRef, ...] = ()


@dataclass(frozen=True)
class ViewCollection:
    sets: tuple[ViewSet, ...]


@dataclass(frozen=True)
class AtlasTile:
    x: int
    y: int
    positions: ArrayRef
    normals: ArrayRef
    face_ids: ArrayRef


@dataclass(frozen=True)
class AtlasContext:
    atlas_id: str
    geometry_id: str
    size: int
    tiles: tuple[AtlasTile, ...]
    chart_ids: ArrayRef
    uv_coverage: ArrayRef
    gbuffer: GBuffer


@dataclass(frozen=True)
class MapInfo:
    atlas_id: str
    semantic: str
    width: int
    height: int
    convention: str = "v_down"
    provenance: str = ""
