"""Exclude unreliable generated RGB only near geometric source boundaries.

The band is measured in pre-upscale camera pixels. No RGB blur or fixed-palette
heuristic is used; alternate cameras and the existing surface fill remain sharp.
"""
import math

def selective_band(depth,normal_ref,source_ref,mask_ref,source_size,margin,torch,_prepared=None,_capture=None,_seed_capture=None,_geometry_capture=None):
    """Reject unsupported RGB near geometric edges using local colour context.

    V3 (margin 2) does not seed rejection from neutral deviations alone. V4
    (margin 6) also catches isolated pale/dark residue with a local consensus.
    Reference colours are robust local medians; no fixed asset palette is used.
    The resulting mask never directly recolours or blurs an RGB pixel.
    """
    import numpy as np
    import torch.nn.functional as F
    from types import SimpleNamespace
    from .surfacefilter import attach
    size=depth.shape[0]
    if source_ref is None or mask_ref is None:
        return torch.zeros((source_size,source_size),device=depth.device,dtype=torch.bool)
    if _prepared is None:
        rgb=np.asarray(source_ref.read())
        if rgb.shape[:2]!=(source_size,source_size):raise ValueError('Source edge repair requires original camera RGB')
        rgb=torch.from_numpy(np.array(rgb[...,:3],np.float32)).to(depth.device)
        if source_ref.read().dtype==np.uint8:rgb/=255.
        fg=torch.from_numpy(np.array(mask_ref.read()>.5)).to(depth.device)
        normal=None
        if normal_ref is not None:
            a=normal_ref.read()
            if a.shape==(size,size,4):normal=torch.from_numpy(np.array(a,np.uint8)).to(depth.device).reshape(-1,4)
        warning=SimpleNamespace(R=size,depth=depth.reshape(-1),surface_normals=normal)
        attach(warning,torch)
        edge=warning.surface_edges[0]
        if size!=source_size:edge=F.adaptive_max_pool2d(edge.float()[None,None],(source_size,source_size))[0,0]>.5
    else:
        rgb,fg,edge=_prepared
    if _capture is not None:_capture.append((rgb,fg,edge))
    radius=max(1,int(math.ceil(margin)))
    edge=F.max_pool2d(edge.float()[None,None],2*radius+1,1,radius)[0,0]>.5
    if _geometry_capture is not None:_geometry_capture.append(edge&fg)
    ids=(edge&fg).nonzero(as_tuple=False);bad=torch.zeros_like(fg)
    # V3 keeps its original two rings; V4 includes wider context for patches.
    # No averaged output or unfolded full-size RGB image is allocated.
    wide=margin>=6
    rings=(4,8,16) if wide else (2,4)
    offsets=[(dy*r,dx*r) for r in rings for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]]
    offsets=torch.tensor(offsets,device=depth.device,dtype=torch.long)
    for start in range(0,len(ids),16384):
        yy,xx=ids[start:start+16384].unbind(1)
        y=(yy[None]+offsets[:,0,None]).clamp(0,source_size-1)
        x=(xx[None]+offsets[:,1,None]).clamp(0,source_size-1)
        ok=fg[y,x];samples=rgb[y,x]
        samples.masked_fill_(~ok[...,None],float('nan'));valid=ok.sum(0)
        median=samples.nanmedian(0).values;centre=rgb[yy,xx]
        a=centre-centre.mean(1,keepdim=True);b=median-median.mean(1,keepdim=True)
        an=a.norm(dim=1);bn=b.norm(dim=1);cosine=(a*b).sum(1)/(an*bn).clamp_min(1e-8)
        colour_difference=(centre-median).norm(dim=1)
        # Coherent coloured details have local support, unlike a rainbow speck
        # or an abrupt hue flip at an undersampled geometric boundary.
        support=((samples-centre[None]).norm(dim=2)<.10).sum(0)
        if wide:
            reject=(valid>=10)&(an>.13)&(bn>.13)&(cosine<.985)&(colour_difference>.12)&(support<12)
        else:
            reject=(valid>=6)&(an>.13)&(bn>.13)&(cosine<.97)&(colour_difference>.14)&(support<8)
        if margin>=6:
            consensus=((samples-median[None]).norm(dim=2)<.15).sum(0)
            reject|=(valid>=12)&(colour_difference>.20)&(support<6)&(consensus>=12)
        bad[yy,xx]=reject
    # Include the upscaler's immediate ringing halo only at these detections.
    halo=5 if wide else 3
    if _seed_capture is not None:_seed_capture.append(bad&edge&fg)
    return (F.max_pool2d(bad.float()[None,None],2*halo+1,1,halo)[0,0]>.5)&edge&fg

def selective_bands(depth,normal_ref,source_ref,mask_ref,source_size,margin,torch,with_seeds=False,with_geometry=False):
    """V3 and V4 decisions share one RGB load and geometric-edge analysis.

    The two detection windows are unchanged. The temporary shared inputs are
    released before building the large enhanced image/mip tensors.
    """
    prepared=[]
    legacy=selective_band(depth,normal_ref,source_ref,mask_ref,source_size,2,torch,_capture=prepared)
    if not prepared:return (legacy,legacy.clone(),legacy.clone()) if with_seeds or with_geometry else (legacy,legacy.clone())
    seeds=[] if with_seeds else None
    geometry=[] if with_geometry else None
    clean=selective_band(depth,normal_ref,source_ref,mask_ref,source_size,margin,torch,_prepared=prepared[0],_seed_capture=seeds,_geometry_capture=geometry)
    return (legacy,clean,geometry[0]) if with_geometry else ((legacy,clean,seeds[0]) if with_seeds else (legacy,clean))

def mask_rows(valid,edges,transform,torch):
    """Apply without a full enhanced-resolution temporary mask."""
    size=valid.shape[0];original=edges.shape[0];device=valid.device
    x=(torch.arange(size,device=device,dtype=torch.float32)+.5)/size
    if transform is not None:
        tr=transform
        x=((x*size-tr.pad_left)/tr.resized_width*tr.width+tr.left)/tr.original_width
    xx=(x*original).floor().long().clamp(0,original-1)
    removed=0
    for start in range(0,size,128):
        end=min(size,start+128)
        y=(torch.arange(start,end,device=device,dtype=torch.float32)+.5)/size
        if transform is not None:
            tr=transform
            y=((y*size-tr.pad_top)/tr.resized_height*tr.height+tr.top)/tr.original_height
        yy=(y*original).floor().long().clamp(0,original-1)
        bad=edges[yy[:,None],xx[None,:]]
        removed+=int(((valid[start:end]>0)&bad).sum())
        valid[start:end].masked_fill_(bad,0)
    return removed
