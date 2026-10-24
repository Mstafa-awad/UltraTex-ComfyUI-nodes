"""Texture-neutral diagnostics of the anchor bake: which camera supplies each texel, and how sure it is.

Enabled only when ``UTX_DIAG_DIR`` names a folder; otherwise every method returns immediately and costs
nothing. Nothing recorded here is read back by the bake, so the texture is identical with or without it.

``diag_layers.npz`` (per primary texel, in raster order; ``texel`` maps them to atlas pixels):
  winner      fallback camera that supplied the colour (0 = anchor view, -1 = unobserved)
  anchor_a    anchor-view weight, q_front / q_fallback / q_best / q_adjusted: observation qualities
  q_by_view   (views, texels) quality of every camera,  counts: cameras that see the texel
``probe.npz`` (``UTX_DIAG_POINT="x,y,z,radius"``): every camera's colour, quality and adjusted quality for the
texels within `radius` of a point of the normalised mesh.
"""
import os

import numpy as np


class AnchorDiagnostics:
    def __init__(self, texels, views, device, torch):
        self.folder = os.environ.get('UTX_DIAG_DIR')
        self.enabled = bool(self.folder)
        self.rows = []
        self.probe = None
        if not self.enabled:
            return
        self.torch, self.device = torch, device
        self.winner = torch.full((texels,), -1, dtype=torch.int8, device=device)
        self.anchor = torch.zeros(texels, device=device)
        self.q_front = torch.zeros(texels, device=device)
        self.q_fallback = torch.zeros(texels, device=device)
        self.q_adjusted = torch.zeros(texels, device=device)
        self.q_by_view = torch.zeros((views, texels), dtype=torch.float16, device=device)
        point = os.environ.get('UTX_DIAG_POINT')
        if point:
            x, y, z, radius = (float(v) for v in point.split(','))
            self.probe = (torch.tensor([x, y, z], device=device), radius)

    def view_quality(self, view, start, stop, q):
        if self.enabled:
            self.q_by_view[view, start:stop] = q.half()

    def fallback_winner(self, view, start, stop, took_lead):
        """`took_lead` is a bool mask of the texels where this view now has the best adjusted quality."""
        if self.enabled:
            self.winner[start:stop] = self.torch.where(took_lead, self.torch.tensor(view, dtype=self.torch.int8, device=self.device),
                                                       self.winner[start:stop])

    def anchor_step(self, start, stop, weight, q_front, q_fallback, q_adjusted):
        if self.enabled:
            self.anchor[start:stop] = weight
            self.q_front[start:stop] = q_front
            self.q_fallback[start:stop] = q_fallback
            self.q_adjusted[start:stop] = q_adjusted
            self.winner[start:stop] = self.torch.where(weight >= 0.5, self.torch.zeros_like(self.winner[start:stop]), self.winner[start:stop])

    def probe_texels(self, start, stop, positions, colour, q, q_weight, weight, fallback):
        if not (self.enabled and self.probe):
            return
        near = ((positions - self.probe[0]).norm(dim=1) < self.probe[1]).nonzero(as_tuple=True)[0]
        if near.numel():
            self.rows.append(dict(texel=(near + start).cpu().numpy(), c=colour[:, near].cpu().numpy(), q=q[:, near].cpu().numpy(),
                                  qw=q_weight[:, near].cpu().numpy(), a=weight[near].cpu().numpy(), before=fallback[near].cpu().numpy()))

    def save(self, q_best, counts, texel):
        if not self.enabled:
            return
        if self.rows:
            np.savez(os.path.join(self.folder, 'probe.npz'),
                     **{k: np.concatenate([r[k] for r in self.rows], 0 if k in ('texel', 'a', 'before') else 1) for k in self.rows[0]})
        np.savez(os.path.join(self.folder, 'diag_layers.npz'), winner=self.winner.cpu().numpy(), anchor_a=self.anchor.cpu().numpy(),
                 q_front=self.q_front.cpu().numpy(), q_fallback=self.q_fallback.cpu().numpy(), q_best=q_best,
                 q_adjusted=self.q_adjusted.cpu().numpy(), q_by_view=self.q_by_view.cpu().numpy(), counts=counts, texel=texel)
