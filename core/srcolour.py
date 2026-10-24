"""Constrain unsupported SR chroma near geometric boundaries, never its detail.

All reference colours come from the same original camera and locally matching
depth/normal surface. No palette, model identity or semantic region is used.
CPU row processing keeps an extra enhanced-resolution image off the GPU.
"""
import threading

import numpy as np


class Guard:
    def __init__(self,band,low,high):
        self.band,self.low,self.high=band,low,high
        self.corrected_pixels=0
        self._lock=threading.Lock()   # rows may be corrected by several threads at once

    def apply(self,rgb,row,canvas,transform):
        size=len(self.band)
        x=(np.arange(canvas,dtype=np.float64)+.5)/canvas
        y=(np.arange(row,row+len(rgb),dtype=np.float64)+.5)/canvas
        if transform is not None:
            t=transform
            x=((x*canvas-t.pad_left)/t.resized_width*t.width+t.left)/t.original_width
            y=((y*canvas-t.pad_top)/t.resized_height*t.height+t.top)/t.original_height
        xx=np.floor(x*size).astype(int).clip(0,size-1);yy=np.floor(y*size).astype(int).clip(0,size-1)
        active=self.band[yy[:,None],xx[None,:]]&(y[:,None]>=0)&(y[:,None]<1)&(x[None,:]>=0)&(x[None,:]<1)
        sy,sx=np.nonzero(active)
        if not len(sy):return
        a=rgb[sy,sx];light=a.mean(1,keepdims=True);chroma=a-light
        lo=self.low[yy[sy],xx[sx]].astype(np.float32);hi=self.high[yy[sy],xx[sx]].astype(np.float32)
        clipped=np.maximum(lo,np.minimum(hi,chroma))
        change=np.linalg.norm(clipped-chroma,axis=1)>.10
        if not change.any():return
        # A channelwise envelope need not have zero mean. Recenter it, then
        # contract only enough to fit RGB gamut while retaining SR mean light.
        c=clipped[change];c-=c.mean(1,keepdims=True);l=light[change]
        pos=c.max(1,keepdims=True);neg=c.min(1,keepdims=True)
        scale=np.minimum(1,np.minimum((1-l)/np.maximum(pos,1e-8),l/np.maximum(-neg,1e-8)))
        rgb[sy[change],sx[change]]=np.clip(l+c*scale,0,1)
        with self._lock:self.corrected_pixels+=int(change.sum())


def prepare(view,depth_t,torch,geometry=None,others=None):
    if view.source_ref is None or view.source_mask_ref is None:return None
    if view.image_ref is view.source_ref or view.image_ref==view.source_ref:return None
    if view.normal_ref is None or view.normal_rotation is None:return None
    source=np.asarray(view.source_ref.read());size=len(source);depth=np.asarray(view.depth_ref.read());R=len(depth)
    normal=np.asarray(view.normal_ref.read())
    if normal.shape!=(R,R,4):return None
    fg=np.asarray(view.source_mask_ref.read())>.5
    if geometry is None:
        from types import SimpleNamespace
        from .surfacefilter import attach
        import torch.nn.functional as F
        warning=SimpleNamespace(R=R,depth=depth_t.reshape(-1),surface_normals=torch.from_numpy(np.array(normal)).to(depth_t.device).reshape(-1,4))
        attach(warning,torch);edge=warning.surface_edges[0]
        if R!=size:edge=F.adaptive_max_pool2d(edge.float()[None,None],(size,size))[0,0]>.5
        geometry=(F.max_pool2d(edge.float()[None,None],13,1,6)[0,0]>.5).cpu().numpy()
    band=np.asarray(geometry,bool)&fg
    low=np.zeros((size,size,3),np.float16);high=np.zeros_like(low)
    ids=np.flatnonzero(band);den=255. if source.dtype==np.uint8 else 1.
    rotation=np.asarray(view.normal_rotation).T@np.asarray(view.matrix)[:3,:3].T
    for start in range(0,len(ids),16384):
        p=ids[start:start+16384];y=p//size;x=p%size;u=(x+.5)/size;v=(y+.5)/size
        iy=np.minimum((v*R).astype(int),R-1);ix=np.minimum((u*R).astype(int),R-1);d=depth[iy,ix]
        n=(normal[iy,ix,:3].astype(np.float32)/127.5-1)
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
        nc=n@rotation;pc=np.stack(((u-.5)/1.5625*d,(.5-v)/1.5625*d,-d),1);num=(nc*pc).sum(1)
        lo=np.full((len(p),3),np.inf,np.float32);hi=-lo
        dark=np.zeros(len(p),bool);light=np.zeros(len(p),bool)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                sx=(x+dx).clip(0,size-1);sy=(y+dy).clip(0,size-1)
                tx=np.minimum(((sx+.5)/size*R).astype(int),R-1);ty=np.minimum(((sy+.5)/size*R).astype(int),R-1)
                nn=normal[ty,tx,:3].astype(np.float32)/127.5-1
                divisor=nc[:,0]*((tx+.5)/R-.5)/1.5625+nc[:,1]*(.5-(ty+.5)/R)/1.5625-nc[:,2]
                predicted=num/np.where(abs(divisor)>.02,divisor,1)
                same=fg[sy,sx]&(depth[ty,tx]>0)&(abs(divisor)>.02)
                same&=abs(depth[ty,tx]-predicted)<.0002+4*d/(1.5625*R)
                same&=(n*nn).sum(1)>.8*np.maximum(np.linalg.norm(nn,axis=1),1e-8)
                rgb=source[sy,sx,:3].astype(np.float32)/den;c=rgb-rgb.mean(1,keepdims=True)
                lo=np.minimum(lo,np.where(same[:,None],c,np.inf));hi=np.maximum(hi,np.where(same[:,None],c,-np.inf))
                dark|=(rgb.max(1)<.18)&fg[sy,sx];light|=(rgb.min(1)>.8)&fg[sy,sx]
        good=np.isfinite(lo).all(1)&~(dark&light)
        band[y,x]&=good
        low[y[good],x[good]]=lo[good];high[y[good],x[good]]=hi[good]
    guard=Guard(band,low,high);guard.consensus_stats={}
    if others is not None:
        from .photoconsistency import confirm
        confirmed,stats,reference=confirm(view,others,torch.from_numpy(band).to(depth_t.device),torch,enhanced_chroma=True,return_reference=True)
        agreed=confirmed.cpu().numpy();c=reference[agreed].astype(np.float32);c-=c.mean(1,keepdims=True)
        # Only the colour unsupported by two independent cameras is bounded.
        # Enhanced mean brightness and supported fine colour remain available.
        low[agreed]=c-.02;high[agreed]=c+.02;guard.consensus_stats=stats
    return guard
