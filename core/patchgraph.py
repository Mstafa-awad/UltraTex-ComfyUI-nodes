"""UV guide neighbours plus links across two-face manifold UV seams."""
import numpy as np

def build(surface,P,A,uvs=None):
    r=surface.raster;top=surface.topology;T=r.size;n=len(r.texel)
    grid=np.full(T*T,-1,np.int32);grid[r.texel]=np.arange(n,dtype=np.int32)
    chart=surface.texel_chart;edges=[]
    def close(i,j):
        return np.linalg.norm(P[i]-P[j],axis=1)<=8*(np.sqrt(A[i])+np.sqrt(A[j]))
    for delta in [1,T]:
        target=r.texel.astype(np.int64)+delta;inside=target<T*T
        if delta==1:inside &= r.texel%T<T-1
        i=np.flatnonzero(inside);j=grid[target[inside]];ok=j>=0;i=i[ok];j=j[ok]
        ok=(chart[i]==chart[j])&close(i,j);edges.append((i[ok],j[ok]))
    # Only mesh edges with duplicated positions can be chart seams. Filtering
    # first avoids allocating/sorting all 18 million face edges of a dense mesh.
    duplicate=np.bincount(top.weld,minlength=top.n_weld)>1
    keys=[];codes=[];f=top.faces;wf=top.weld_faces
    for start in range(0,len(f),131072):
        ff=f[start:start+131072];ww=wf[start:start+131072]
        for side in range(3):
            nxt=(side+1)%3;keep=duplicate[top.weld[ff[:,side]]]&duplicate[top.weld[ff[:,nxt]]]
            ids=np.flatnonzero(keep);a=ww[ids,side];b=ww[ids,nxt]
            keys.append(np.minimum(a,b).astype(np.int64)*top.n_weld+np.maximum(a,b))
            codes.append((ids.astype(np.int64)+start)*3+side)
    seam_count=0
    if keys:
        key=np.concatenate(keys);code=np.concatenate(codes);del keys,codes
        order=np.argsort(key,kind='stable');key=key[order];code=code[order];del order
        first=np.r_[True,key[1:]!=key[:-1]];starts=np.flatnonzero(first);counts=np.diff(np.r_[starts,len(key)])
        starts=starts[counts==2];a=code[starts];b=code[starts+1];del key,code
        fa=a//3;fb=b//3;keep=top.face_chart[fa]!=top.face_chart[fb];a=a[keep];b=b[keep];fa=fa[keep];fb=fb[keep]
        def lookup(code):
            face=code//3;side=code%3;uv=surface.uvs if uvs is None else uvs
            mid=(uv[f[face,side]]+uv[f[face,(side+1)%3]])*.5*T-.5
            target_chart=top.face_chart[face];cx=np.round(mid[:,0]).astype(int);cy=np.round(mid[:,1]).astype(int)
            best=np.full(len(mid),np.inf);node=np.full(len(mid),-1,np.int32)
            for dy in range(-3,4):
                for dx in range(-3,4):
                    x=cx+dx;y=cy+dy;inside=(x>=0)&(x<T)&(y>=0)&(y<T);ix=np.flatnonzero(inside)
                    cand=grid[y[ix]*T+x[ix]];good=cand>=0;ix=ix[good];cand=cand[good]
                    good=chart[cand]==target_chart[ix];ix=ix[good];cand=cand[good]
                    d=(x[ix]-mid[ix,0])**2+(y[ix]-mid[ix,1])**2;good=d<best[ix];ix=ix[good];cand=cand[good]
                    node[ix]=cand;best[ix]=d[good]
            return node
        i=lookup(a);j=lookup(b);valid=(i>=0)&(j>=0)&(i!=j);i=i[valid];j=j[valid]
        good=close(i,j);i=i[good];j=j[good];seam_count=len(i);edges.append((i,j))
    ii=np.concatenate([e[0] for e in edges]);jj=np.concatenate([e[1] for e in edges]);lo=np.minimum(ii,jj);hi=np.maximum(ii,jj)
    key=lo.astype(np.int64)*n+hi;_,pick=np.unique(key,return_index=True)
    return lo[pick].astype(np.int32),hi[pick].astype(np.int32),dict(manifold_seam_links_before_dedup=seam_count,edges=len(pick))
