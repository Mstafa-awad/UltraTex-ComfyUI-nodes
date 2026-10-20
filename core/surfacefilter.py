"""Sparse geometry-aware reconstruction where an ordinary mip crosses surfaces.

The normal path is unchanged. At an identified geometric boundary, RGB taps
are separately tested against the query triangle plane and shading normal.
"""
def attach(view,torch):
    """Small bool mip pyramid of geometric edges; bounded-row normal decoding."""
    import torch.nn.functional as F
    R=view.R;d=view.depth.reshape(R,R);fg=d>0;edge=torch.zeros_like(fg)
    normal=getattr(view,'surface_normals',None)
    if normal is not None:normal=normal.reshape(R,R,4)
    for row in range(0,R,64):
        stop=min(R,row+64)
        a=d[row:stop,:-1];b=d[row:stop,1:];jump=(a>0)!=(b>0)
        if normal is not None:
            n=normal[row:stop,:,:3].float()/127.5-1
            jump|=((n[:,:-1]*n[:,1:]).sum(-1)<.65)&(a>0)&(b>0)
        edge[row:stop,:-1]|=jump;edge[row:stop,1:]|=jump
        last=min(stop,R-1)
        if row<last:
            a=d[row:last];b=d[row+1:last+1];jump=(a>0)!=(b>0)
            if normal is not None:
                aN=normal[row:last,:,:3].float()/127.5-1;bN=normal[row+1:last+1,:,:3].float()/127.5-1
                jump|=((aN*bN).sum(-1)<.65)&(a>0)&(b>0)
            edge[row:last]|=jump;edge[row+1:last+1]|=jump
        mid=d[row:stop,1:-1];curvature=(d[row:stop,:-2]+d[row:stop,2:]-2*mid).abs()
        edge[row:stop,1:-1]|=(curvature>1e-5+.25*mid/(1.5625*R))&(mid>0)
        a=max(1,row);b=min(stop,R-1)
        if a<b:
            mid=d[a:b];curvature=(d[a-1:b-1]+d[a+1:b+1]-2*mid).abs()
            edge[a:b]|=(curvature>1e-5+.25*mid/(1.5625*R))&(mid>0)
    levels=[edge]
    while levels[-1].shape[0]>1 and levels[-1].shape[0]%2==0:
        levels.append(F.max_pool2d(levels[-1].float()[None,None],2)[0,0]>0)
    view.surface_edges=levels

def sample(view,Pc,Ng,Ns,u0,v0,u,v,footprint,ordinary,torch,q_hint=None):
    R=view.R;S=view.S;tr=view.transform
    nc=Ng@view.rot.T;num=(nc*Pc).sum(1);z=-Pc[:,2]
    tolerance=1e-5+.35*z/(1.5625*R)
    normal=None
    if getattr(view,'surface_normals',None) is not None:
        normal=Ns@view.surface_normal_rotation
    def matching(uu,vv,sel=None):
        x=(uu*R-.5).round().long().clamp(0,R-1);y=(vv*R-.5).round().long().clamp(0,R-1)
        D=view.depth[y*R+x]
        nn=nc if sel is None else nc[sel];nnum=num if sel is None else num[sel];zz=z if sel is None else z[sel];tol=tolerance if sel is None else tolerance[sel]
        den=nn[:,0]*((x.float()+.5)/R-.5)/1.5625+nn[:,1]*(.5-(y.float()+.5)/R)/1.5625-nn[:,2]
        steep=den.abs()>.02;t=torch.where(steep,nnum/torch.where(steep,den,torch.ones_like(den)),zz)
        good=(D>0)&((D-t).abs()<=tol)
        if normal is not None:
            data=view.surface_normals[y*R+x];n=data[:,:3].float()/127.5-1
            target=normal if sel is None else normal[sel]
            same=(n*target).sum(1)>=.65*n.norm(dim=1).clamp_min(1e-8)
            good &= same | (data[:,3]==0)
        return good
    if tr is None:
        sx=sy=float(S)
    else:
        sx=tr.original_width*tr.resized_width/tr.width;sy=tr.original_height*tr.resized_height/tr.height
    spread=(footprint.square()/8-.25).clamp_min(0).sqrt()
    radius=torch.maximum(1+spread,torch.full_like(footprint,1.5))
    clean=matching(u0,v0)
    # The pooled cell width covers the complete reconstruction window, so its
    # four corner cells conservatively include every intervening edge pixel.
    if getattr(view,'exact_filter_support',False):
        from .filtersupport import edge_overlap
        clean &= ~edge_overlap(view,u,v,footprint,torch)
    else:
        lod=torch.ceil(torch.log2(2*radius*max(R/sx,R/sy)+1)).long().clamp(0,len(view.surface_edges)-1)
        for level,edges in enumerate(view.surface_edges):
            ids=(lod==level).nonzero(as_tuple=True)[0]
            if not ids.numel():continue
            size=edges.shape[0]
            for dx,dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                xx=((u0[ids]+dx*radius[ids]/sx)*size).long().clamp(0,size-1)
                yy=((v0[ids]+dy*radius[ids]/sy)*size).long().clamp(0,size-1)
                clean[ids]&=~edges[yy,xx]
    if q_hint is not None:clean |= q_hint<.06
    selected=(~clean).nonzero(as_tuple=True)[0]
    if not selected.numel():return ordinary,~clean
    uu=u[selected];vv=v[selected];step=spread[selected]
    total=torch.zeros_like(ordinary[selected]);data=view.levels[0][0];channels=data.shape[0]
    # Nine subpixel boxes; mask normalisation clips the filter to this surface.
    offsets=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    for ox,oy in offsets:
        px=uu*S-.5+ox*step;py=vv*S-.5+oy*step
        ix=px.floor();iy=py.floor();fx=px-ix;fy=py-iy
        for dx in [0,1]:
            for dy in [0,1]:
                x=(ix+dx).long().clamp(0,S-1);y=(iy+dy).long().clamp(0,S-1)
                eu=(x.float()+.5)/S;ev=(y.float()+.5)/S
                if tr is None:ou=eu;ov=ev
                else:
                    ou=((eu*S-tr.pad_left)/tr.resized_width*tr.width+tr.left)/tr.original_width
                    ov=((ev*S-tr.pad_top)/tr.resized_height*tr.height+tr.top)/tr.original_height
                support=matching(ou,ov,selected)
                w=(fx if dx else 1-fx)*(fy if dy else 1-fy)*support/9
                total+=data[:,y,x].T*w[:,None]
    ordinary[selected]=total
    view.guard_candidates=getattr(view,'guard_candidates',0)+int(selected.numel())
    view.guard_rejected=getattr(view,'guard_rejected',0)+int((total[:,-1]<=.05).sum())
    return ordinary,~clean
