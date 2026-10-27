"""Bilinear border agreement by changing atlas padding only.

Only padding texels in the bilinear support of matched UV borders are variables. Three
point Gaussian quadrature integrates the squared bilinear mismatch exactly on
each interval between grid crossings. Interior texture samples are fixed.
No mesh, UV, normal or occupied surface sample is changed. Albedo is solved in
linear light after sRGB decoding; scalar material maps use linear values directly. Independent
NumPy/SciPy implementation; no external seam-erasure source code is incorporated.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import LinearOperator,cg
from . import fastbake as fb

def paired_edges(context,topology=None):
    top=topology or fb.build_topology(context.vertices,context.faces,context.uvs)
    f=top.faces;wf=top.weld_faces;uv=fb.uv_vdown(context)
    if not np.isfinite(uv).all() or uv.min() < -1e-4 or uv.max()>1+1e-4:
        raise ValueError('UV border repair requires finite UV coordinates within [0,1]')
    codes=np.arange(len(f)*3);face=codes//3;side=codes%3
    a=wf[face,side];b=wf[face,(side+1)%3]
    key=np.minimum(a,b).astype(np.int64)*top.n_weld+np.maximum(a,b)
    valid=a!=b;key=key[valid];codes=codes[valid]
    order=np.argsort(key,kind='stable');key=key[order];codes=codes[order]
    if not len(key):return np.empty((0,4,2),np.float64),top
    first=np.flatnonzero(np.r_[True,key[1:]!=key[:-1]])
    counts=np.diff(np.r_[first,len(key)]);first=first[counts==2]
    ca,cb=codes[first],codes[first+1];fa,sa=ca//3,ca%3;fb_,sb=cb//3,cb%3
    ia,ja=f[fa,sa],f[fa,(sa+1)%3];ib,jb=f[fb_,sb],f[fb_,(sb+1)%3]
    reverse=top.weld[ia]!=top.weld[ib]
    ib,jb=np.where(reverse,jb,ib),np.where(reverse,ib,jb)
    seam=(np.linalg.norm(uv[ia]-uv[ib],axis=1)>1e-6)|(np.linalg.norm(uv[ja]-uv[jb],axis=1)>1e-6)
    # Authored hard-normal borders can intentionally separate materials.
    na=context.normals[ia]+context.normals[ja];nb=context.normals[ib]+context.normals[jb]
    cosine=(na*nb).sum(1)/np.maximum(np.linalg.norm(na,axis=1)*np.linalg.norm(nb,axis=1),1e-12)
    seam&=cosine>.7
    seam&=(top.face_area[fa]>1e-16)&(top.face_area[fb_]>1e-16)
    return np.stack([uv[ia[seam]],uv[ja[seam]],uv[ib[seam]],uv[jb[seam]]],1),top

def _quadrature(points):
    cuts=[np.array([0.,1.])]
    for a,b in zip(points[[0,2]].ravel(),points[[1,3]].ravel()):
        if abs(b-a)<1e-12:continue
        grid=np.arange(np.floor(min(a,b))+1,np.ceil(max(a,b)))
        t=(grid-a)/(b-a);cuts.append(t[(t>1e-10)&(t<1-1e-10)])
    t=np.unique(np.concatenate(cuts));half=(t[1:]-t[:-1])*.5;mid=(t[1:]+t[:-1])*.5
    keep=half>1e-10;half=half[keep];mid=mid[keep]
    point=(mid[:,None]+half[:,None]*np.array([-np.sqrt(3/5),0,np.sqrt(3/5)])).ravel()
    weight=(half[:,None]*np.array([5/9,8/9,5/9])).ravel()
    return point,weight

def _bilinear_taps(points,size):
    points=np.clip(points,0,size-1)
    xy=np.floor(points).astype(np.int32);fraction=points-xy
    x,y=xy[:,0],xy[:,1];fx,fy=fraction[:,0],fraction[:,1]
    xx=np.minimum(x+1,size-1);yy=np.minimum(y+1,size-1)
    index=np.stack([y*size+x,y*size+xx,yy*size+x,yy*size+xx],1)
    weight=np.stack([(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy],1)
    return index,weight

def build_system(context,size,topology=None,selection=None):
    started=time.perf_counter();edges,top=paired_edges(context,topology)
    if selection is not None:edges=edges[np.asarray(selection,bool)]
    columns=[];coefficients=[];samples=0
    for ei,edge in enumerate(edges):
        if ei%256==0:fb._interrupt()
        pixel=edge*size-.5;t,w=_quadrature(pixel)
        a=pixel[0,None]*(1-t[:,None])+pixel[1,None]*t[:,None]
        b=pixel[2,None]*(1-t[:,None])+pixel[3,None]*t[:,None]
        ia,wa=_bilinear_taps(a,size);ib,wb=_bilinear_taps(b,size)
        length=max(np.linalg.norm(pixel[1]-pixel[0]),np.linalg.norm(pixel[3]-pixel[2]),1.)
        weight=np.sqrt(w*length)[:,None]
        columns.append(np.concatenate([ia,ib],1).astype(np.int32))
        coefficients.append((np.concatenate([wa,-wb],1)*weight).astype(np.float32));samples+=len(t)
    if not columns:return None
    columns=np.concatenate(columns);data=np.concatenate(coefficients)
    unknown=np.unique(columns)
    lookup=np.full(size*size,-1,np.int32);lookup[unknown]=np.arange(len(unknown),dtype=np.int32)
    row=np.repeat(np.arange(samples,dtype=np.int32),8)
    matrix=coo_matrix((data.ravel(),(row,lookup[columns.ravel()])),shape=(samples,len(unknown))).tocsr()
    matrix.sum_duplicates();matrix.eliminate_zeros()
    return matrix,unknown,dict(seam_edges=len(edges),integrated_samples=samples,unknown_texels=len(unknown),
        matrix_nonzeros=matrix.nnz,assembly_seconds=time.perf_counter()-started)

def repair_padding(context,atlas,strength=100.,system=None,inplace=False,color_space='srgb'):
    started=time.perf_counter();image=np.asarray(atlas)
    if image.ndim!=3 or image.shape[2]!=3 or image.shape[0]!=image.shape[1]:raise ValueError('Expected square RGB atlas')
    if color_space not in ('srgb','linear'):raise ValueError('Expected srgb or linear color space')
    if image.dtype!=np.uint8 and image.dtype.kind!='f':raise ValueError('Expected uint8 or floating point atlas')
    if image.dtype!=np.uint8 and (not np.isfinite(image).all() or image.min()<0 or image.max()>1):
        raise ValueError('UV border repair requires finite RGB values in [0,1]')
    if not np.isfinite(strength):raise ValueError('UV border repair strength must be finite')
    def unchanged():
        return (image if inplace else image.copy()),dict(changed_texels=0,changed_padding_texels=0,
            occupied_surface_texels_unchanged=True,color_space=color_space,seconds=time.perf_counter()-started)
    if strength<=0:return unchanged()
    size=image.shape[0];system=system or build_system(context,size)
    if system is None:return unchanged()
    matrix,unknown,report=system;report=dict(report)
    surface,_=fb.prepare_surface(context,size)
    occupied=np.zeros(size*size,bool);occupied[surface.raster.texel]=True
    interior=occupied[unknown]
    source=image.reshape(-1,3)[unknown].astype(np.float32)/(255 if image.dtype==np.uint8 else 1.)
    linear=fb.srgb_to_linear_np(source) if color_space=='srgb' else source.copy()
    residual=matrix@linear
    free=~interior
    if not free.any():return unchanged()
    selected=matrix[:,free].tocsr()
    fidelity=np.full(int(free.sum()),.05,np.float32)
    square=np.asarray(selected.power(2).sum(0)).ravel()
    diagonal=fidelity+strength*square
    transpose=selected.T.tocsr()
    operator=LinearOperator((int(free.sum()),int(free.sum())),
        matvec=lambda x:fidelity*x+strength*(transpose@(selected@x)),dtype=np.float32)
    preconditioner=LinearOperator(operator.shape,matvec=lambda x:x/diagonal,dtype=np.float32)
    corrected=linear.copy();status=[];iterations=[]
    for c in range(3):
        if c and np.array_equal(linear[:,c],linear[:,c-1]):
            corrected[:,c]=corrected[:,c-1];status.append(status[-1]);iterations.append(0);continue
        count=[0]
        def callback(_):
            count[0]+=1
            if count[0]%16==0:fb._interrupt()
        delta,info=cg(operator,-strength*(transpose@residual[:,c]),M=preconditioner,
            rtol=2e-5,atol=1e-8,maxiter=400,callback=callback)
        corrected[free,c]+=delta;status.append(int(info));iterations.append(count[0])
    corrected=np.clip(corrected,0,1)
    if color_space=='srgb':corrected=fb.linear_to_srgb_np(corrected)
    result=image if inplace else image.copy()
    if image.dtype==np.uint8:
        values=np.uint8(np.clip(corrected*255+.5,0,255));written=values.astype(np.float32)/255
    else:values=corrected.astype(image.dtype);written=values.astype(np.float32)
    values[~free]=image.reshape(-1,3)[unknown[~free]]
    written[~free]=source[~free]
    changed=int(np.any(values!=image.reshape(-1,3)[unknown],axis=1).sum())
    target=unknown[free]
    result[target//size,target%size]=values[free]
    after=matrix@(fb.srgb_to_linear_np(written) if color_space=='srgb' else written)
    difference=np.abs(written-source)
    report.update(strength=strength,padding_fidelity=.05,color_space=color_space,
        solver_status=status,iterations=iterations,linear_seam_rms_before=float(np.sqrt(np.mean(residual**2))),
        linear_seam_rms_after=float(np.sqrt(np.mean(after**2))),
        modified_support_fraction=len(unknown)/(size*size),
        interior_rgb_difference_mean=float(difference[interior].mean()) if interior.any() else 0.,
        interior_rgb_difference_p99=float(np.quantile(difference[interior],.99)) if interior.any() else 0.,
        changed_padding_texels=changed,occupied_surface_texels_unchanged=True,optimized_texels=int(free.sum()),
        correction_scope='padding beside paired UV borders only; all occupied surface pixels exactly unchanged',
        seconds=time.perf_counter()-started)
    return result,report
