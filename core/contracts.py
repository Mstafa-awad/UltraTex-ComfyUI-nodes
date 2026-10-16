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
class ViewSet:
    plan: ViewPlan
    images: tuple[ArrayRef, ...]
    masks: tuple[ArrayRef, ...]
    task: str
    job_id: str
    gbuffer: GBuffer


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
