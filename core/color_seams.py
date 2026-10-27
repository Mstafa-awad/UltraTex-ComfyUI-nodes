"""Independent global seam-leveling with smooth split-vertex log-color offsets."""
import time
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg,LinearOperator
from scipy.ndimage import gaussian_filter,map_coordinates
from . import fastbake as fb

def level(context,atlas,radius=2.,strength=20.,regularization=4.,limit=1.35,inplace=False):
    start=time.perf_counter();image=np.asarray(atlas)
    if image.ndim!=3 or image.shape[2]!=3 or image.shape[0]!=image.shape[1]:raise ValueError('Expected square RGB atlas')
    if image.dtype.kind!='f' or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
        raise ValueError('Color leveling requires finite floating point RGB in [0,1]')
    if not all(np.isfinite(x) for x in (radius,strength,regularization,limit)) or radius<=0 or regularization<0 or limit<1:
        raise ValueError('Invalid color leveling parameters')
    if strength<=0 or len(context.vertices)>500000:
        return image if inplace else image.copy(),dict(applied=False,reason='disabled' if strength<=0 else 'dense mesh budget',fine_detail_filtering=False)
    size=len(atlas);surface,_=fb.prepare_surface(context,size);top=surface.topology
    f=top.faces;wf=top.weld_faces;uv=fb.uv_vdown(context);nv=len(context.vertices)
    code=np.arange(len(f)*3);face=code//3;side=code%3
    a=wf[face,side];b=wf[face,(side+1)%3];key=np.minimum(a,b).astype(np.int64)*top.n_weld+np.maximum(a,b)
    order=np.argsort(key,kind='stable');key=key[order];code=code[order]
    first=np.flatnonzero(np.r_[True,key[1:]!=key[:-1]]);count=np.diff(np.r_[first,len(key)])
    first=first[count==2];ca=code[first];cb=code[first+1]
    fa,sa=ca//3,ca%3;fj,sj=cb//3,cb%3
    ia,ja=f[fa,sa],f[fa,(sa+1)%3];ib,jb=f[fj,sj],f[fj,(sj+1)%3]
    reverse=top.weld[ia]!=top.weld[ib];ib,jb=np.where(reverse,jb,ib),np.where(reverse,ib,jb)
    na=context.normals[ia]+context.normals[ja];nb=context.normals[ib]+context.normals[jb]
    dot=(na*nb).sum(1)/np.maximum(np.linalg.norm(na,axis=1)*np.linalg.norm(nb,axis=1),1e-12)
    keep=((np.linalg.norm(uv[ia]-uv[ib],axis=1)>1e-6)|(np.linalg.norm(uv[ja]-uv[jb],axis=1)>1e-6))&(dot>.8)
    ia,ja,ib,jb,fa,fj,sa,sj=[x[keep] for x in (ia,ja,ib,jb,fa,fj,sa,sj)]
    if not len(ia):return np.asarray(atlas).copy(),dict(seam_edges=0,changed=False,solver_status=[0,0,0])
    t=np.array([.1,.3,.5,.7,.9],np.float64);weights=np.stack([1-t,t],1)
    def positions(i,j,face,side):
        p=uv[i,None]*(1-t[None,:,None])+uv[j,None]*t[None,:,None]
        opposite=uv[f[face,(side+2)%3]]
        edge=uv[j]-uv[i];edge/=np.maximum(np.linalg.norm(edge,axis=1,keepdims=True),1e-12)
        inward=opposite[:,None]-p;inward-=edge[:,None]*(inward*edge[:,None]).sum(2)[...,None]
        length=np.linalg.norm(inward,axis=2,keepdims=True)
        return (p+inward/np.maximum(length,1e-12)*np.minimum(radius/size,length*.15))*size-.5
    pa=positions(ia,ja,fa,sa);pb=positions(ib,jb,fj,sj)
    left=[];right=[]
    for channel in range(3):
        fb._interrupt()
        scalar=fb.srgb_to_linear_np(np.asarray(atlas[...,channel],np.float32))
        low=gaussian_filter(scalar,1.5)
        left.append(map_coordinates(low,[pa[...,1],pa[...,0]],order=1,mode='nearest'))
        right.append(map_coordinates(low,[pb[...,1],pb[...,0]],order=1,mode='nearest'))
        del scalar,low
    left,right=np.stack(left,-1),np.stack(right,-1)
    delta=np.log(np.maximum(left,.008))-np.log(np.maximum(right,.008))
    distance=np.linalg.norm(np.sqrt(np.maximum(left,0))-np.sqrt(np.maximum(right,0)),axis=2)
    valid=(distance<.35)&(left.max(2)>.015)&(right.max(2)>.015)
    if not valid.any():return np.asarray(atlas).copy(),dict(seam_edges=len(ia),changed=False,solver_status=[0,0,0])
    row=np.repeat(np.arange(len(ia)*len(t)),4)
    col=np.stack([np.repeat(ia,5),np.repeat(ja,5),np.repeat(ib,5),np.repeat(jb,5)],1).ravel()
    data=np.tile(np.stack([1-t,t,-(1-t),-t],1),(len(ia),1))*valid.reshape(-1,1)
    A=coo_matrix((data.ravel(),(row,col)),shape=(len(ia)*5,nv)).tocsr()
    ii=f[:,[0,1,2]].ravel();jj=f[:,[1,2,0]].ravel()
    W=coo_matrix((np.ones(len(ii)*2),(np.r_[ii,jj],np.r_[jj,ii])),shape=(nv,nv)).tocsr()
    degree=np.asarray(W.sum(1)).ravel();L=(diags(degree)-W)/max(np.median(degree[degree>0]),1)
    matrix=(regularization*L+diags(np.full(nv,.05))+strength*(A.T@A)).tocsr()
    inv=1/matrix.diagonal();M=LinearOperator(matrix.shape,matvec=lambda x:x*inv,dtype=np.float64)
    field=np.zeros((nv,3));status=[]
    for c in range(3):
        iteration=[0]
        def callback(_):
            iteration[0]+=1
            if iteration[0]%20==0:fb._interrupt()
        value,info=cg(matrix,-strength*(A.T@(delta[...,c].ravel()*valid.ravel())),M=M,rtol=1e-5,maxiter=1200,callback=callback)
        field[:,c]=value.clip(-np.log(limit),np.log(limit));status.append(int(info))
    if any(status):return image if inplace else image.copy(),dict(applied=False,solver_status=status,reason='solver did not converge',fine_detail_filtering=False)
    result=np.asarray(atlas,np.float32) if inplace else np.asarray(atlas,np.float32).copy()
    flat=result.reshape(-1,3);r=surface.raster
    for begin in range(0,len(r.texel),1<<18):
        end=min(begin+(1<<18),len(r.texel));ids=r.texel[begin:end]
        gain=np.exp((field[f[r.face[begin:end]]]*r.bary[begin:end,...,None]).sum(1))
        flat[ids]=fb.linear_to_srgb_np(np.clip(fb.srgb_to_linear_np(flat[ids])*gain,0,1))
    gutter=surface.gutter_source()
    for begin in range(0,len(flat),1<<18):
        end=min(begin+(1<<18),len(flat));flat[begin:end]=flat[gutter[begin:end]]
    residual=(delta.reshape(-1,3)+(A@field))[valid.ravel()]
    return result,dict(applied=True,solver_status=status,seam_edges=len(ia),log_seam_rms_before=float(np.sqrt(np.mean(delta[valid]**2))),
        log_seam_rms_after=float(np.sqrt(np.mean(residual**2))),max_gain=limit,seconds=time.perf_counter()-start,
        fine_detail_filtering=False,correction='smooth split-vertex color gains; original texel detail retained')
