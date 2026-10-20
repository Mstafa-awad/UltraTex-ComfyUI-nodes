"""Optional spatially coherent source selection and bounded seam correction."""
from dataclasses import dataclass
import gc,time,json,dataclasses,hashlib
import numpy as np
from . import fastbake as fb
from . import patchgraph,patchcut

GUIDE_SIZE=2048
PREFERENCE=4.

def cache_key(context,collection,size):
    from .runtime import fingerprint
    records=[]
    for v in fb.collect_views(collection):
        records.append((v.image_ref.digest,v.mask_ref.digest,v.depth_ref.digest,
                        None if v.normal_ref is None else v.normal_ref.digest,v.matrix.tolist(),
                        None if v.transform is None else dataclasses.asdict(v.transform)))
    from .enhance import restore_original
    original=[(v.image_ref.digest,v.mask_ref.digest) for v in fb.collect_views(restore_original(collection))]
    return fingerprint('coherent-guide-v5-original-surface-filter',context.geometry_id,context.atlas_id,int(size),records,original)

def cached_guide(context,collection,size,policy,guide=None):
    """Cache exact arrays with the host Store; no pickle or arbitrary paths."""
    from .runtime import Store,atomic_json
    start=time.perf_counter();store=Store(policy);key=cache_key(context,collection,size)
    meta=store.root/('coherent-guide-'+key+'.json');names=['label','chart','correction','gains']
    if guide is None:
        try:
            if not meta.exists() or meta.stat().st_size>2**20:return None
            report=json.loads(meta.read_text(encoding='utf-8'))
            if report['version']!=1:return None
            refs=[store.lookup(key+':'+name) for name in names]
            if any(ref is None for ref in refs):return None
            arrays=[ref.read() for ref in refs];K=len(fb.collect_views(collection))
            expected=[((size,size),'u'),((size,size),'i'),((size,size,3),'f'),((K,3),'f')]
            if any(a.shape!=shape or a.dtype.kind!=kind for a,(shape,kind) in zip(arrays,expected)):return None
            r=dict(report['report'],cache_hit=True,seconds=time.perf_counter()-start)
            return Guide(*arrays,r)
        except (KeyError,ValueError,OSError,EOFError):return None
    try:
        for name in names:store.put(getattr(guide,name),key+':'+name)
        atomic_json(meta,dict(version=1,report=guide.report));guide.report['cache_write_seconds']=time.perf_counter()-start
    except (MemoryError,OSError) as error:
        guide.report['cache_write_failed']=str(error)
    guide.report['cache_hit']=False
    return guide

@dataclass
class Guide:
    label: object
    chart: object
    correction: object
    gains: object
    report: dict

def build_guide(surface,collection,device,uvs=None):
    from .enhance import restore_original
    from .sharpbake import proxy_view
    torch=fb._torch();start=time.perf_counter();r=surface.raster;n=len(r.texel)
    geo=fb.texel_geometry(surface,device,analytic_footprint=True)
    originals=fb.collect_views(restore_original(collection));originals=[proxy_view(v) if v.image_ref.shape[0]>2048 else v for v in originals]
    gains,gain_report=fb.estimate_gains(geo,originals,device,complete_mips=True)
    # Selection and low-frequency correction need the preserved original images,
    # not a second expensive decode/downsample of ten enhanced float canvases.
    # Final RGB reconstruction below still uses every full enhanced source.
    views=originals;C=np.empty((len(views),n,3),np.float32);Q=np.empty((len(views),n),np.float32)
    with torch.inference_mode():
        for vi,v in enumerate(views):
            fb._interrupt();proxy=proxy_view(v,2048);pv=fb.prepare_view(proxy,'base_color',device,complete_mips=True,stream=True,surface_guard=True)
            for s in range(0,n,1<<18):
                e=min(n,s+(1<<18));c,q=fb._project(geo.slice(s,e),pv,torch)
                C[vi,s:e]=c.cpu().numpy()*gains[vi];Q[vi,s:e]=q.cpu().numpy()
            del pv,proxy
    sample_seconds=time.perf_counter()-start
    P=geo.P[:n].cpu().numpy();A=geo.A[:n].cpu().numpy();del geo
    ii,jj,graph_report=patchgraph.build(surface,P,A,uvs);del P,A
    active=Q.max(0)>=fb.Q_LO;ids=np.flatnonzero(active);lookup=np.full(n,-1,np.int32);lookup[ids]=np.arange(len(ids),dtype=np.int32)
    keep=active[ii]&active[jj];i=lookup[ii[keep]];j=lookup[jj[keep]];del ii,jj,lookup
    label=np.full(n,255,np.uint8);correction=np.zeros((n,3),np.float32)
    cut_report={};level_report={}
    if len(ids):
        cs=C[:,ids];qs=Q[:,ids];del C,Q
        labels,cut_report=patchcut.solve(cs,qs,i,j,smoothness=4.,epochs=2,log=lambda _:fb._interrupt())
        _,corr,level_report=patchcut.level_seams(cs,qs,labels,i,j,screen=.02,max_gain=1.3)
        label[ids]=labels;correction[ids]=corr
    T=r.size;lab=np.full(T*T,255,np.uint8);charts=np.zeros(T*T,np.int32);corr=np.zeros((T*T,3),np.float32)
    lab[r.texel]=label;charts[r.texel]=surface.texel_chart;corr[r.texel]=correction
    report=dict(guide_size=T,guide_nodes=n,observed_guide_nodes=len(ids),sampling_seconds=sample_seconds,guide_images='preserved originals; final RGB uses enhanced views',
                graph=graph_report,cut=cut_report,seam_leveling=level_report,
                gain_matching=gain_report,seconds=time.perf_counter()-start)
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return Guide(lab.reshape(T,T),charts.reshape(T,T),corr.reshape(T,T,3),gains,report)

def preference_map(surface,guide):
    """Nearest guide support from the SAME UV chart; no global gutter transfer."""
    T=surface.raster.size;G=guide.label.shape[0];all_t=surface.raster.texel
    label=np.full(len(all_t),255,np.uint8);corr=np.zeros((len(all_t),3),np.float32)
    for s in range(0,len(all_t),1<<19):
        e=min(len(all_t),s+(1<<19));t=all_t[s:e];chart=surface.texel_chart[s:e]
        x=(t%T).astype(np.float32);y=(t//T).astype(np.float32);x=(x+.5)*(G/T)-.5;y=(y+.5)*(G/T)-.5
        cx=np.floor(x).astype(np.int32);cy=np.floor(y).astype(np.int32);best=np.full(e-s,np.inf,np.float32);labs=label[s:e]
        for dy in [0,1]:
            for dx in [0,1]:
                xx=np.clip(cx+dx,0,G-1);yy=np.clip(cy+dy,0,G-1)
                valid=(guide.chart[yy,xx]==chart)&(guide.label[yy,xx]!=255)
                distance=(x-xx)**2+(y-yy)**2;choose=valid&(distance<best)
                best[choose]=distance[choose];labs[choose]=guide.label[yy[choose],xx[choose]]
        den=np.zeros(e-s,np.float32);num=np.zeros((e-s,3),np.float32)
        for dy in [0,1]:
            for dx in [0,1]:
                xx=np.clip(cx+dx,0,G-1);yy=np.clip(cy+dy,0,G-1)
                valid=(guide.chart[yy,xx]==chart)&(guide.label[yy,xx]==labs)&(labs!=255)
                w=(1-abs(x-(cx+dx)))*(1-abs(y-(cy+dy)))*valid
                num+=guide.correction[yy,xx]*w[:,None];den+=w
        corr[s:e]=num/np.maximum(den,1e-8)[:,None]
    return label,corr

def bake(surface,collection,device,guide,completion=True,timer=None):
    from .sharpbake import attach_source_base
    torch=fb._torch();timer=timer or fb.StageTimer();geo=fb.texel_geometry(surface,device,analytic_footprint=True);n=len(surface.raster.texel)
    labels,corr=preference_map(surface,guide)
    # Keep guide preference/correction on CPU and stream chunks, avoiding an
    # extra full fine-atlas GPU allocation. Fewer accumulators than sharp_source.
    full=torch.zeros((n,3),device=device,dtype=torch.float32);weight=torch.zeros(n,device=device,dtype=torch.float32);select_max=torch.zeros(n,device=device,dtype=torch.float32)
    qmax=torch.zeros(n,device=device,dtype=torch.float32);counts=torch.zeros(n,dtype=torch.int16,device=device);per=[];chunk=1<<18
    timer.lap('base_color:coherent_geometry')
    for vi,v in enumerate(fb.collect_views(collection)):
        fb._interrupt();pv=attach_source_base(fb.prepare_view(v,'base_color',device,complete_mips=True,stream=True,surface_guard=True));g=torch.tensor(guide.gains[vi],device=device,dtype=torch.float32);used=0
        for start in range(0,len(geo.P),chunk):
            end=min(len(geo.P),start+chunk)
            # Primary and overlapping claims must not share a slice boundary.
            ranges=[(start,min(end,n),False),(max(start,n),end,True)]
            for s,e,extra in ranges:
                if e<=s:continue
                c,q,_=fb._project(geo.slice(s,e),pv,torch,source_detail=True);c*=g
                pos=geo.extra_pos[s-n:e-n] if extra else slice(s,e)
                indices=surface.raster.extra_pos[s-n:e-n] if extra else slice(s,e)
                desired=(torch.from_numpy((labels[indices]==vi)).to(device)&(q>=fb.Q_LO)) if not extra else torch.zeros_like(q,dtype=torch.bool)
                correction=torch.from_numpy(corr[indices]).to(device);c*=torch.where(desired[:,None],correction.exp(),torch.ones_like(correction))
                selection=q*torch.where(desired,PREFERENCE,1.)
                if not extra:
                    old=select_max[pos];new=torch.maximum(old,selection);scale=(old/new.clamp_min(1e-20)).pow(64);w=(selection/new.clamp_min(1e-20)).pow(64)
                    full[pos]=full[pos]*scale[:,None]+c*w[:,None];weight[pos]=weight[pos]*scale+w;select_max[pos]=new
                    qmax[pos]=torch.maximum(qmax[pos],q);counts[pos]+=(q>=fb.Q_LO).to(torch.int16);used+=int((q>=fb.Q_LO).sum())
                else:
                    unique=torch.unique(pos);old=select_max[unique].clone();select_max.scatter_reduce_(0,pos,selection,reduce='amax',include_self=True)
                    scale=(old/select_max[unique].clamp_min(1e-20)).pow(64);full[unique]*=scale[:,None];weight[unique]*=scale
                    w=(selection/select_max[pos].clamp_min(1e-20)).pow(64);full.index_add_(0,pos,c*w[:,None]);weight.index_add_(0,pos,w)
                    qmax.scatter_reduce_(0,pos,q,reduce='amax',include_self=True)
        per.append(dict(view=pv.label,texels_used=used,image_px=pv.S,depth_px=pv.R,guard_candidates=getattr(pv,'guard_candidates',0),guard_rejected=getattr(pv,'guard_rejected',0)));del pv
    full/=weight.clamp_min(1e-20)[:,None];colour=np.nan_to_num(full.cpu().numpy(),nan=0,posinf=1,neginf=0);q=qmax.cpu().numpy();counts=counts.cpu().numpy()
    del geo,full,weight,select_max,qmax;timer.lap('base_color:coherent_projection')
    alpha=fb._smoothstep(q,fb.Q_LO,fb.Q_HI);area=surface.texel_area.astype(np.float64)
    stats=dict(texels=n,observed_strong_fraction=float((alpha>=1-1e-6).mean()),observed_any_fraction=float((q>=fb.Q_LO).mean()),
        unobserved_fraction=float((q<fb.Q_LO).mean()),observed_area_fraction=float((area*alpha).sum()/max(area.sum(),1e-30)),
        mean_views_per_observed_texel=float(counts[q>=fb.Q_LO].mean()) if (q>=fb.Q_LO).any() else 0,
        fusion=dict(mode='sharp_coherent',preference_factor=PREFERENCE,guide=guide.report,
                    fine_guide_support_fraction=float((labels!=255).mean()),source_images_warped=False),views=per,view_gains_rgb=guide.gains.tolist())
    if completion:values,stats['completion']=fb.complete_texels(surface,colour,alpha,device)
    else:values=np.where((q>=fb.Q_LO)[:,None],colour,0)
    timer.lap('base_color:completion')
    return np.clip(fb.linear_to_srgb_np(values),0,1),alpha,stats
