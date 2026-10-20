"""Source-calibrated colour/detail separation, independent NumPy/PyTorch implementation.

No checkpoints, source-image warps, new UVs or external reconstruction packages.
Only this optional branch uses these calculations; the retained bakers are unchanged.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import math
import numpy as np
from . import fastbake as fb

SOURCE_RADIUS = 8.0           # sigma in ORIGINAL camera pixels, before enhancement
SOURCE_POWER = 64.0           # narrow, continuous detail handover
COLOUR_GAIN_LIMIT = .10       # multiplicative broad colour adjustment; black stays black


@dataclass
class _MemoryRef:
    array: object
    @property
    def shape(self): return self.array.shape
    def read(self): return self.array


def proxy_view(view, cap=1024):
    """Bounded-memory linear-light area proxy for colour gain estimation only.

    Read rows from the mmap instead of materialising an 8K RGB copy. Its affine
    stays exact even when crop padding is fractional at the proxy resolution.
    Proxies never become output images or change persisted crop metadata.
    """
    image = view.image_ref.read(); mask = view.mask_ref.read(); S = image.shape[0]
    k = 1
    while S//k > cap and S % (2*k) == 0: k *= 2
    if k == 1: return view
    s = S//k; rgb = np.empty((s,s,3),np.float32); m = np.empty((s,s),np.float32)
    for y in range(0,s,32):
        e=min(s,y+32)
        a=np.asarray(image[y*k:e*k],np.float32)
        if image.dtype==np.uint8: a=a/255.
        a=fb.srgb_to_linear_np(a)
        mm=np.asarray(mask[y*k:e*k],np.float32)
        premult=a*mm[...,None]
        den=mm.reshape(e-y,k,s,k).mean((1,3))
        num=premult.reshape(e-y,k,s,k,3).mean((1,3))
        rgb[y:e]=fb.linear_to_srgb_np(num/np.maximum(den[...,None],1e-6))
        m[y:e]=den
    tr=view.transform
    if tr is not None:
        fields={key:getattr(tr,key) for key in ('original_width','original_height','left','top','width','height',
               'output_size','resized_width','resized_height','pad_left','pad_top')}
        tr=SimpleNamespace(**fields,uv=tr.uv,proxy_affine=True)
        tr.output_size=s
        for name in ('resized_width','resized_height','pad_left','pad_top'): setattr(tr,name,getattr(tr,name)/k)
    return replace(view,image_ref=_MemoryRef(rgb),mask_ref=_MemoryRef(m),transform=tr)


def proxy_edge_mask(edge,tr,torch):
    """Resample original edge exclusions with the proxy's exact fractional affine."""
    import torch.nn.functional as F
    s=tr.output_size
    x=torch.arange(s,device=edge.device,dtype=torch.float32)+.5
    u=((x-tr.pad_left)/tr.resized_width*tr.width+tr.left)/tr.original_width
    v=((x-tr.pad_top)/tr.resized_height*tr.height+tr.top)/tr.original_height
    yy,xx=torch.meshgrid(v,u,indexing='ij')
    grid=torch.stack((xx*2-1,yy*2-1),-1)[None]
    return F.grid_sample(edge.float()[None,None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,0]>.01


def attach_source_base(view):
    """A physically fixed detail band across 2K, cropped 4K and cropped 8K views."""
    import torch.nn.functional as F
    level=view.levels[0]
    for candidate in view.levels:
        level=candidate
        if candidate.shape[-1]<=512: break
    if level.shape[-1]>512:  # odd original crop: no regular mip chain
        level=F.interpolate(level,size=(512,512),mode='area')
    scale=1.
    if view.transform is not None:
        tr=view.transform
        scale=math.sqrt(tr.resized_width/tr.width*tr.resized_height/tr.height)
    sigma=max(.35,SOURCE_RADIUS*scale*level.shape[-1]/view.S)
    view.source_base=fb.blur_premultiplied(level,sigma).contiguous()
    return view


def sample_source_detail(view,u,v,colour,lod,q,torch,cubic_allowed=None):
    """Small base sample + bounded cubic reconstruction without full-size helper maps."""
    import torch.nn.functional as F
    grid=torch.stack((2*u-1,2*v-1),-1).view(1,1,-1,2)
    low=F.grid_sample(view.source_base,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
    x=(u*view.S-.5).floor().long(); y=(v*view.S-.5).floor().long()
    need=(lod<=fb.DETAIL_LOD_MAX)&(q>=fb.Q_LO)&(x>=1)&(y>=1)&(x<view.S-2)&(y<view.S-2)
    if cubic_allowed is not None:need &= cubic_allowed
    ix=need.nonzero(as_tuple=True)[0]
    if ix.numel():
        # Require the complete cubic support to be foreground, including depth
        # discontinuity exclusions. The mask is sampled, never eroded globally.
        a=view.levels[0][0,-1];xx=x[ix];yy=y[ix];ok=torch.ones_like(xx,dtype=torch.bool)
        for dy in (-1,0,1,2):
            for dx in (-1,0,1,2): ok &= a[yy+dy,xx+dx]>.999
        ix=ix[ok]
        if ix.numel():
            xx=x[ix];yy=y[ix];data=view.levels[0][0,:3]
            corners=torch.stack([data[:,yy+dy,xx+dx].T for dy,dx in ((0,0),(0,1),(1,0),(1,1))])
            cub=F.grid_sample(view.levels[0][:,:3],grid[:,:,ix],mode='bicubic',padding_mode='border',align_corners=False)[0,:,0].T
            colour[ix]=cub.clamp(corners.amin(0),corners.amax(0))
    return colour,low


def _normalised_add(acc,idx,colour,low,q,torch):
    """Stable relative powers; also supports repeated overlapping-UV targets."""
    full,base,broad,weight,bweight,qmax=acc
    if isinstance(idx,slice):
        old=qmax[idx];new=torch.maximum(old,q)
        scale=(old/new.clamp_min(1e-20)).pow(SOURCE_POWER)
        w=(q/new.clamp_min(1e-20)).pow(SOURCE_POWER)
        full[idx]=full[idx]*scale[:,None]+colour*w[:,None]
        base[idx]=base[idx]*scale[:,None]+low*w[:,None]
        weight[idx]=weight[idx]*scale+w
        wb=q.square();broad[idx]+=low*wb[:,None];bweight[idx]+=wb;qmax[idx]=new
    else:
        unique=torch.unique(idx);old=qmax[unique].clone()
        qmax.scatter_reduce_(0,idx,q,reduce='amax',include_self=True)
        scale=(old/qmax[unique].clamp_min(1e-20)).pow(SOURCE_POWER)
        full[unique]*=scale[:,None];base[unique]*=scale[:,None];weight[unique]*=scale
        w=(q/qmax[idx].clamp_min(1e-20)).pow(SOURCE_POWER);wb=q.square()
        full.index_add_(0,idx,colour*w[:,None]);base.index_add_(0,idx,low*w[:,None])
        broad.index_add_(0,idx,low*wb[:,None]);weight.index_add_(0,idx,w);bweight.index_add_(0,idx,wb)


def fuse_source_views(geo,views,device,gains=None,chunk=1<<19):
    torch=fb._torch();n=geo.n_primary if geo.n_primary>=0 else len(geo.P)
    def zeros(c=None): return torch.zeros((n,c) if c else n,dtype=torch.float32,device=device)
    # full/base/broad, narrow/broad weights, best geometric quality.
    acc=(zeros(3),zeros(3),zeros(3),zeros(),zeros(),zeros())
    counts=torch.zeros(n,dtype=torch.int16,device=device);per=[]
    for vi,v in enumerate(views):
        fb._interrupt();pv=attach_source_base(fb.prepare_view(v,'base_color',device,complete_mips=True,stream=True))
        g=None if gains is None else torch.tensor(gains[vi],dtype=torch.float32,device=device)
        used=0
        def project_range(start,stop,extra=False):
            nonlocal used
            for s in range(start,stop,chunk):
                e=min(stop,s+chunk);c,q,low=fb._project(geo.slice(s,e),pv,torch,source_detail=True)
                if g is not None:c=c*g;low=low*g
                ix=geo.extra_pos[s-n:e-n] if extra else slice(s,e)
                _normalised_add(acc,ix,c,low,q,torch)
                if not extra:
                    good=q>=fb.Q_LO;counts[s:e]+=good.to(torch.int16);used+=int(good.sum())
        project_range(0,n)
        if len(geo.P)>n:project_range(n,len(geo.P),True)
        per.append(dict(view=pv.label,texels_used=used,depth_px=pv.R,image_px=pv.S,
                        colour_base_px=int(pv.source_base.shape[-1])))
        del pv
    full,base,broad,w,wb,qmax=acc
    full.div_(w.clamp_min(1e-20)[:,None]);base.div_(w.clamp_min(1e-20)[:,None]);broad.div_(wb.clamp_min(1e-20)[:,None])
    ratio=colour_ratio(base,broad)
    correction=full*(ratio-1)
    full.mul_(ratio)
    return full,qmax,counts,per,dict(mean_abs_colour_correction=float(correction.abs().mean()),
        mean_dominant_detail_weight=float((1/w.clamp_min(1)).mean()))


def colour_ratio(base,broad):
    """Bounded exposure/chroma transfer, never an additive lift of a dark feature."""
    return ((broad+.01)/(base+.01)).clamp(1-COLOUR_GAIN_LIMIT,1+COLOUR_GAIN_LIMIT)


def bake_source_sharp(surface,collection,device,completion=True,harmonize=True,timer=None):
    torch=fb._torch();timer=timer or fb.StageTimer();views=fb.collect_views(collection)
    geo=fb.texel_geometry(surface,device,analytic_footprint=True)
    if device.type=='cuda':torch.cuda.synchronize(device)
    timer.lap('base_color:texel_geometry')
    gains=None;gain_report={}
    if harmonize and len(views)>1:
        from .enhance import restore_original
        original_views=fb.collect_views(restore_original(collection))
        # Estimate only the broad exposure correction from the preserved input
        # views. This avoids a second 8K image preparation pass. AI RGB remains
        # the source of BOTH final detail and its coarse colour base.
        matching_views=[proxy_view(v) if v.image_ref.shape[0]>2048 else v for v in original_views]
        gains,gain_report=fb.estimate_gains(geo,matching_views,device,complete_mips=True)
        if any(a is not b for a,b in zip(original_views,matching_views)):
            gain_report.pop('sampling_density',None) # proxies are not source-density evidence
        gain_report['gain_images']='preserved original views; proxies only above 2048px'
        gain_report['matching_sizes']=[v.image_ref.shape[0] for v in matching_views]
        del original_views,matching_views
        timer.lap('base_color:original_colour_matching')
    colour,qmax,nviews,per,extra=fuse_source_views(geo,views,device,gains)
    if device.type=='cuda':torch.cuda.synchronize(device)
    timer.lap('base_color:source_projection_fusion')
    colour=np.nan_to_num(colour.cpu().numpy(),nan=0.,posinf=1.,neginf=0.)
    qmax=qmax.cpu().numpy();nviews=nviews.cpu().numpy();del geo
    colour=np.clip(colour,0,None).astype(np.float32);alpha=fb._smoothstep(qmax,fb.Q_LO,fb.Q_HI)
    area=surface.texel_area.astype(np.float64)
    stats=dict(texels=int(len(alpha)),observed_strong_fraction=float((alpha>=1-1e-6).mean()),
        observed_any_fraction=float((qmax>=fb.Q_LO).mean()),unobserved_fraction=float((qmax<fb.Q_LO).mean()),
        observed_area_fraction=float((area*alpha).sum()/max(area.sum(),1e-30)),
        mean_views_per_observed_texel=float(nviews[qmax>=fb.Q_LO].mean()) if (qmax>=fb.Q_LO).any() else 0.,
        fusion=dict(mode='sharp_source',source_radius_original_camera_px=SOURCE_RADIUS,detail_power=SOURCE_POWER,
            broad_power=2.,colour_gain_limit=COLOUR_GAIN_LIMIT,
            footprint='world/UV differential; full mask-normalised mip pyramid',
            reconstruction='bounded cubic under magnification; trilinear under minification',
            source_images_warped=False,**extra),views=per)
    if gains is not None:stats.update(view_gains_rgb=gains.tolist(),harmonisation=gain_report)
    if completion:values,stats['completion']=fb.complete_texels(surface,colour,alpha,device)
    else:values=np.where((qmax>=fb.Q_LO)[:,None],colour,0).astype(np.float32)
    timer.lap('base_color:completion')
    return np.clip(fb.linear_to_srgb_np(values),0,1),alpha,stats
