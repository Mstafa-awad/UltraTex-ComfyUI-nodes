"""Protect front-visible source samples from stronger edge repair geometrically."""
import numpy as np

def visible_source(depth, matrix, front, size, torch,source_view=None,candidates=None):
    """Conservative original-camera visibility, with bounded row temporaries.

    A generous depth envelope and expanded support protect transition details.
    Over-protection retains v3 paint; no semantic face or asset palette is used.
    """
    import torch.nn.functional as F
    original=np.asarray(depth.detach().cpu(),np.float32)
    ref=np.asarray(front.depth_ref.read(),np.float32)
    r=len(original);fr=len(ref);m=np.asarray(matrix,np.float64);fm=np.asarray(front.matrix,np.float64)
    inverse=np.linalg.inv(m);result=np.zeros((size,size),bool)
    source_normal=front_normal=None
    if source_view is not None and source_view.normal_ref is not None and front.normal_ref is not None:
        if source_view.normal_rotation is not None and front.normal_rotation is not None:
            source_normal=np.asarray(source_view.normal_ref.read());front_normal=np.asarray(front.normal_ref.read())
    # Only rejected RGB needs a visibility decision. Expand by the protection
    # halo so sparse and full-grid evaluation give the same answer there.
    if candidates is None:
        ids=np.arange(size*size,dtype=np.int64)
    else:
        required=F.max_pool2d(candidates.float()[None,None],5,1,2)[0,0]>.5
        ids=np.flatnonzero(required.cpu().numpy());del required
    for start in range(0,len(ids),32768):
        points=ids[start:start+32768];y=(points//size+.5)/size;x=(points%size+.5)/size
        iy=np.minimum((y*r).astype(np.int64),r-1);ix=np.minimum((x*r).astype(np.int64),r-1)
        d=original[iy,ix]
        pc=np.stack(((x-.5)/1.5625*d,(.5-y)/1.5625*d,-d),-1)
        world=pc@inverse[:3,:3].T+inverse[:3,3]
        p=world@fm[:3,:3].T+fm[:3,3];z=-p[...,2]
        u=.5+1.5625*p[...,0]/np.maximum(z,1e-8)
        v=.5-1.5625*p[...,1]/np.maximum(z,1e-8)
        ok=(d>0)&(z>.05)&(u>=0)&(u<=1)&(v>=0)&(v<=1)
        xx=np.floor(u*fr-.5).astype(np.int64);yy=np.floor(v*fr-.5).astype(np.int64)
        visible=np.zeros(d.shape,bool)
        normal=None
        if source_normal is not None:
            normal=(source_normal[iy,ix,:3].astype(np.float32)/127.5-1)@np.asarray(source_view.normal_rotation).T
        tolerance=.0002+4*z/(1.5625*fr)
        for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
            fd=ref[(yy+dy).clip(0,fr-1),(xx+dx).clip(0,fr-1)]
            same=(fd>0)&(abs(fd-z)<=tolerance)
            if normal is not None:
                fn=(front_normal[(yy+dy).clip(0,fr-1),(xx+dx).clip(0,fr-1),:3].astype(np.float32)/127.5-1)@np.asarray(front.normal_rotation).T
                same&=(normal*fn).sum(-1)>.8*np.linalg.norm(normal,axis=-1)*np.linalg.norm(fn,axis=-1)
            visible|=same
        result.reshape(-1)[points]=visible&ok
    protected=torch.from_numpy(result).to(depth.device)
    return F.max_pool2d(protected.float()[None,None],5,1,2)[0,0]>.5
