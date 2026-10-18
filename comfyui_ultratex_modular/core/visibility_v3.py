"""Visibility Projection v3: subpixel-aware conservative depth + bounded triangle fallback.

Independent NumPy implementation; no OpenMVS/AGPL dependency.
Depth units are mesh camera-space Z (not perspective depth-buffer values).
"""
from __future__ import annotations
import numpy as np


def subpixel_visibility(z, u, v, depth, base_tolerance=0.003, max_tolerance=0.025):
    """Return (visible, uncertain, quality, status) for camera-space points.

    Bilinear only on a continuous depth surface. At depth discontinuities use
    nearest SUPPORTING raster samples; reject obvious foreground occluders.
    status: 0 outside, 1 visible, 2 uncertain near discontinuity, 3 depth mismatch.
    """
    z = np.asarray(z, np.float32).reshape(-1)
    u = np.asarray(u, np.float32).reshape(-1)
    v = np.asarray(v, np.float32).reshape(-1)
    d = np.asarray(depth)
    h,w=d.shape
    good=(z > .05)&np.isfinite(z)&np.isfinite(u)&np.isfinite(v)&(u>=0)&(u<1)&(v>=0)&(v<1)
    px=u*w-.5; py=v*h-.5
    x0=np.clip(np.floor(px).astype(np.int32),0,w-1)
    y0=np.clip(np.floor(py).astype(np.int32),0,h-1)
    x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    fx=np.clip(px-x0,0,1);fy=np.clip(py-y0,0,1)
    d00=d[y0,x0];d10=d[y0,x1];d01=d[y1,x0];d11=d[y1,x1]
    ds=np.stack((d00,d10,d01,d11),axis=1)
    valid=ds>0
    clean=np.where(valid,ds,np.inf)
    lo=clean.min(axis=1)
    hi=np.where(valid,ds,-np.inf).max(axis=1)
    span=np.maximum(hi-lo,0)
    support=valid.sum(axis=1)
    # A surface must be continuous within an approximately 1% depth gradient.
    continuous=good&(support==4)&(span<=np.maximum(.008,.012*z))
    interp=(d00*(1-fx)*(1-fy)+d10*fx*(1-fy)+d01*(1-fx)*fy+d11*fx*fy)
    tolerance=np.minimum(float(max_tolerance),float(base_tolerance)+.0015*z+.65*span)
    smooth_match=continuous&(np.abs(z-interp)<=tolerance)
    # Mixed-depth neighborhoods cannot be safely interpolated across occluders.
    # A matching support pixel is only a *candidate* for an exact triangle check.
    borderline=good & ~continuous & (support>0) & (np.abs(ds-z[:,None])<=np.maximum(.006,.004*z)[:,None]).any(axis=1)
    visible=smooth_match
    uncertain=good&~visible&((support>0)&((np.abs(z-lo)<=float(max_tolerance)*2)|(borderline)))
    status=np.where(good,3,0).astype(np.uint8)
    status[uncertain]=2
    status[visible]=1
    quality=np.zeros(len(z),np.float32)
    quality[visible]=np.clip(1-np.abs(z[visible]-interp[visible])/np.maximum(tolerance[visible],1e-6),.35,1.)
    return visible,uncertain,quality,status


def triangle_ray_fallback(points, camera_center, candidate_ids, vertices, faces, depth_epsilon=0.006):
    """Exact segment/triangle intersections for bounded ambiguous camera pixels.

    candidate_ids are the *frontmost* face IDs from a camera raster's 2x2 or
    3x3 neighborhood. All four neighbors are checked rather than trusting a
    rounded pixel. Does not claim global ray-casting when candidates are absent.
    """
    points=np.asarray(points,np.float32)
    if not len(points):return np.zeros(0,bool)
    tri=np.asarray(candidate_ids,np.int64)
    if tri.ndim==1:tri=tri[:,None]
    o=np.asarray(camera_center,np.float32)
    direction=points-o
    target_distance=np.linalg.norm(direction,axis=1)
    direction/=np.maximum(target_distance[:,None],1e-9)
    nearest=np.full(len(points),np.inf,np.float32)
    verts=np.asarray(vertices,np.float32)
    index=np.asarray(faces,np.int64)
    nfaces=len(index)
    for k in range(tri.shape[1]):
        ids=tri[:,k]
        valid=(ids>=0)&(ids<nfaces)
        if not valid.any():continue
        selection=np.flatnonzero(valid)
        triangle=verts[index[ids[selection]]]
        a=triangle[:,0]; e1=triangle[:,1]-a; e2=triangle[:,2]-a
        rays=direction[selection]
        h=np.cross(rays,e2)
        det=np.einsum('ij,ij->i',e1,h)
        mask=np.abs(det)>1e-9
        inv=1/np.where(mask,det,1)
        s=o-a
        ub=np.einsum('ij,ij->i',s,h)*inv
        q=np.cross(s,e1)
        vb=np.einsum('ij,ij->i',rays,q)*inv
        t=np.einsum('ij,ij->i',e2,q)*inv
        hits=mask&(ub>=-1e-5)&(vb>=-1e-5)&(ub+vb<=1+1e-5)&(t>0)
        near=np.where(hits,t,np.inf).astype(np.float32)
        nearest[selection]=np.minimum(nearest[selection],near)
    hit=np.isfinite(nearest)
    # Occluder in front of the target is a true rejection. If visible candidate
    # triangles intersect exactly at the target depth, this is recoverable.
    return hit&(np.abs(nearest-target_distance)<=float(depth_epsilon))


def camera_face_neighbors(face_id_map, u, v):
    face=np.asarray(face_id_map)
    h,w=face.shape
    px=np.asarray(u)*w-.5;py=np.asarray(v)*h-.5
    x=np.floor(px).astype(np.int32);y=np.floor(py).astype(np.int32)
    xs=np.stack((np.clip(x,0,w-1),np.clip(x+1,0,w-1),np.clip(x,0,w-1),np.clip(x+1,0,w-1)),axis=1)
    ys=np.stack((np.clip(y,0,h-1),np.clip(y,0,h-1),np.clip(y+1,0,h-1),np.clip(y+1,0,h-1)),axis=1)
    return np.asarray(face[ys,xs],np.int32)


def diagnostic_preview(status, uv_cover, size=512):
    """Preview: green projected, red depth rejection, orange mask, purple facing,
    blue offscreen, black UV-unmapped. Status evaluated before fill/repair.
    """
    from PIL import Image
    palette=np.asarray([(0,0,0),(48,205,92),(255,64,64),(255,174,34),(164,64,255),(64,118,255)],np.uint8)
    idx=np.asarray(status,np.uint8).copy()
    idx=np.clip(idx,0,len(palette)-1)
    img=palette[idx]
    img[~np.asarray(uv_cover,bool)]=0
    if max(img.shape[:2])>int(size):
        scale=float(size)/max(img.shape[:2]);w=int(img.shape[1]*scale);h=int(img.shape[0]*scale)
        img=np.asarray(Image.fromarray(img).resize((w,h),Image.Resampling.NEAREST))
    return img.astype(np.float32)/255.0
