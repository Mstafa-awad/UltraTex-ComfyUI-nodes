"""Reproject a baked first-pass atlas into the second calibrated camera rig.

This is a lightweight CPU operation: it does NOT use diffusion weights or
change source UVs. Output init views are 6 images in the order expected by
Flux2 UltraTex, with per-pixel preservation confidence for SDEdit.
"""
from __future__ import annotations

import numpy as np

from .bake import bilinear
from .render import Renderer
from .runtime import interrupt


def reproject_atlas(mesh_context, gbuffer, atlas_rgb, observed, confidence,
                    min_confidence=0.35):
    size = int(gbuffer.plan.sample_size)
    th, tw = atlas_rgb.shape[:2]
    if th != tw or observed.shape != (th, tw) or confidence.shape != (th, tw):
        raise ValueError("Reprojection: atlas RGB, seen and confidence must be square and aligned")
    images = np.ones((6, size, size, 3), np.float32)
    keep = np.zeros((6, size, size), np.float32)
    # Camera raster stores interpolated UVs. Interpolation is perspective
    # correct because OpenGL performs it in projective screen space.
    with Renderer(mesh_context) as renderer:
        for vi, matrix in enumerate(gbuffer.plan.matrices):
            interrupt()
            if vi not in getattr(gbuffer.plan, "active_indices", tuple(range(6))):
                continue  # no 45-degree top/bottom reprojection for 10-view mode
            _nd, uvf = renderer.draw(size, size, matrix, return_uv=True)
            visible = uvf[..., 3] > 0
            if not visible.any():
                continue
            uv = uvf[visible, :2]
            x = uv[:, 0] * tw - 0.5
            y = (1.0 - uv[:, 1]) * th - 0.5
            colors = bilinear(atlas_rgb, x, y).astype(np.float32)
            direct = bilinear(observed, x, y)[:, 0]
            conf = bilinear(confidence, x, y)[:, 0]
            protected = np.clip((conf - min_confidence) / max(1-min_confidence,1e-4),0.,1.)
            # The source must have real observations. Sampling outside the
            # observed atlas never converts a repaired guess into hard guidance.
            protected *= np.clip((direct - .8) / .2,0.,1.)
            images[vi][visible] = colors
            keep[vi][visible] = protected
    return np.clip(images,0.,1.), keep
