from __future__ import annotations

import numpy as np

from .contracts import AtlasContext, AtlasTile, MapInfo
from .geometry import FOCAL
from .render import Renderer
from .runtime import Store, fingerprint, interrupt, measure, ram_preflight


def srgb_to_linear(x):
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055)**2.4)


def linear_to_srgb(x):
    x = np.maximum(x, 0)
    return np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4) - .055)


def chart_labels(context):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    faces = context.faces
    # UV seams are split vertices in native MESH; connectivity stays within each island.
    rows = np.concatenate((faces[:, 0], faces[:, 1]))
    cols = np.concatenate((faces[:, 1], faces[:, 2]))
    edges = coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(len(context.vertices),)*2).tocsr()
    _, labels = connected_components(edges, directed=False)
    return labels[faces[:, 0]].astype(np.int32) + 1


def prepare_atlas(context, gbuffer, size, policy):
    if context.geometry_id != gbuffer.plan.geometry_id:
        raise ValueError("G-buffer and mesh geometry do not match")
    if size < 64 or size > 8192:
        raise ValueError("Atlas size must be between 64 and 8192")
    if len(context.faces) >= 2**24:
        raise ValueError("Atlas rasterizer supports fewer than 16,777,216 faces")
    # UV overlap/degeneracy validation intentionally disabled by request.
    ram_preflight(size*size*8 + len(context.faces)*32, policy.system_headroom_mb)
    store = Store(policy)
    # New cache namespace: this version rasterizes without an overlap-counting pass.
    prefix = fingerprint("atlas-raster-v5-no-uv-validation", context.atlas_id, context.normals, size, policy.atlas_tile)
    chart_ref = store.lookup(prefix + "charts")
    cover_ref = store.lookup(prefix + "cover")
    tiles = []
    for y in range(0, size, policy.atlas_tile):
        for x in range(0, size, policy.atlas_tile):
            refs = [store.lookup(f"{prefix}:{x}:{y}:{what}") for what in ("pos", "nrm", "faces")]
            tiles.append((x, y, refs))
    if chart_ref and cover_ref and all(all(r is not None for r in refs) for _, _, refs in tiles):
        return AtlasContext(context.atlas_id, context.geometry_id, size,
                            tuple(AtlasTile(x,y,*refs) for x,y,refs in tiles), chart_ref, cover_ref, gbuffer), {
                                "stage":"prepare_atlas", "cache_hit":True, "atlas_size":size,
                                "uv_validation":"disabled by user request"}
    with measure("prepare_atlas", atlas_size=size) as report:
        per_face = chart_labels(context)
        # Do not reject UV island overlaps, stacked faces, or collapsed UV triangles.
        # Draw once per tile; the renderer will select whichever overlapping face
        # is visible in the UV rasterization at each texel.
        charts = np.zeros((size,size), np.int32)
        complete = []
        with Renderer(context) as renderer:
            for x,y,_ in tiles:
                interrupt()
                w,h = min(policy.atlas_tile, size-x), min(policy.atlas_tile,size-y)
                region = (x/size, 1-(y+h)/size, w/size, h/size)
                nd,pf = renderer.draw(w,h,np.eye(4),region)
                face_ids = pf[...,3].astype(np.int32)-1
                valid = face_ids >= 0
                labels = charts[y:y+h,x:x+w]
                labels[valid] = per_face[face_ids[valid]]
                refs = [store.put(data,f"{prefix}:{x}:{y}:{what}") for data,what in
                        ((pf[...,:3],"pos"),(nd[...,:3],"nrm"),(face_ids,"faces"))]
                complete.append(AtlasTile(x,y,*refs))
        chart_ref,cover_ref = store.put(charts,prefix+"charts"),store.put(charts>0,prefix+"cover")
        report.update(cache_hit=False, uv_coverage=float((charts>0).mean()), tiles=len(complete),
                      uv_validation="disabled by user request")
    return AtlasContext(context.atlas_id, context.geometry_id, size,tuple(complete),chart_ref,cover_ref,gbuffer),report


def bilinear(image, x, y):
    h,w = image.shape[:2]
    x,y = np.clip(x,0,w-1),np.clip(y,0,h-1)
    x0,y0 = x.astype(np.int32),y.astype(np.int32)
    x1,y1 = np.minimum(x0+1,w-1),np.minimum(y0+1,h-1)
    fx,fy = (x-x0)[:,None],(y-y0)[:,None]
    if image.ndim == 2:
        image = image[...,None]
    return ((image[y0,x0]*(1-fx)+image[y0,x1]*fx)*(1-fy)+
            (image[y1,x0]*(1-fx)+image[y1,x1]*fx)*fy)


def valid_view_mask(rgb, mask, white_rim=12.):
    """Discard generated white rims only within the silhouette's narrow border."""
    from scipy.ndimage import distance_transform_edt
    mask=np.asarray(mask,np.float32).copy()
    if white_rim>0 and np.any(mask):
        threshold=230 if rgb.dtype==np.uint8 else .9
        near=distance_transform_edt(mask>.5)<=white_rim*mask.shape[0]/1024.
        mask[near & (rgb[...,:3]>threshold).all(axis=-1)]=0
    return mask


def projection_ramp(depth, mask, edge_pixels, edge_jump=.01):
    """Feather silhouettes AND internal depth discontinuities (hands, folds, face)."""
    from scipy.ndimage import distance_transform_edt
    from PIL import Image
    cover=depth>0
    if not cover.any():
        return np.zeros(depth.shape,np.float32)
    scale=depth.shape[0]/mask.shape[0]
    if mask.shape!=depth.shape:
        mask=np.asarray(Image.fromarray(mask).resize(depth.shape[::-1],Image.Resampling.NEAREST))
    gy,gx=np.gradient(np.where(cover,depth,depth[cover].max()+1.))
    edges=(~cover)|(mask<.5)|(np.hypot(gx,gy)>edge_jump)
    if edge_pixels<=0:
        return (~edges).astype(np.float32)
    return np.minimum(distance_transform_edt(~edges)/max(edge_pixels*scale,1e-6),1).astype(np.float32)



def bake_map(atlas, views, semantic, policy, power=4., depth_tolerance=.004,
             edge_pixels=6., best_view_mix=.5, view_indices=(0,1,2,3,4,5), original=None,gutter=8,
             blend_mode="detail_preserving", extra_views=(), visibility_mode="hybrid",
             geometry_context=None, visibility_diagnostics=False):
    if atlas.geometry_id != views.plan.geometry_id or atlas.gbuffer.plan.identity != views.plan.identity:
        raise ValueError("Atlas and views must use the same calibrated geometry/view plan")
    if semantic in ("roughness","metallic") and views.task not in ("mr",semantic,"external"):
        raise ValueError(f"Cannot interpret {views.task} views as {semantic}")
    if semantic == "base_color" and views.task == "mr":
        raise ValueError("MR views are scalar data, not base color")
    if not view_indices or any(i<0 or i>=6 for i in view_indices):
        raise ValueError("Select at least one valid view index 0..5")
    if blend_mode not in ("smooth", "detail_preserving"):
        raise ValueError("blend_mode must be detail_preserving or smooth")
    if visibility_mode not in ("hybrid", "adaptive_depth"):
        raise ValueError("Visibility mode must be hybrid or adaptive_depth")
    from .visibility_v3 import subpixel_visibility, triangle_ray_fallback, camera_face_neighbors, diagnostic_preview
    from .render import get_camera_face_ids
    sources=[(views,tuple(view_indices))]
    for extra, selected in extra_views:
        if extra.plan.geometry_id!=atlas.geometry_id or extra.task!=views.task:
            raise ValueError("Additional views must use the same geometry and texture task")
        if extra.gbuffer.plan.identity!=extra.plan.identity:
            raise ValueError("Additional views have inconsistent camera calibration")
        if not selected or any(i<0 or i>=6 for i in selected):
            raise ValueError("Additional view indices must be 0..5")
        sources.append((extra,tuple(selected)))
    t=atlas.size
    # Gutter EDT adds distance/index buffers; account for its temporary storage too.
    ram_preflight(t*t*(64 if gutter>0 else 36) + views.plan.sample_size**2*40,policy.system_headroom_mb)
    color = semantic in ("base_color","emissive")
    output=np.zeros((t,t,3),np.float32)
    confidence=np.zeros((t,t),np.float32)
    seen=np.zeros((t,t),bool)
    reasons=np.zeros((t,t),np.uint8) if visibility_diagnostics else None
    recovered_rays=0
    rejected_depth=0
    missing_mask=0
    # Source evidence is disk backed; projection accumulators are bounded to one UV tile.
    store=Store(policy)
    with measure("bake_map",semantic=semantic,atlas_size=t) as report:
        prepared={}
        # Transform each source view only once, not once per UV tile. Subsequent lookups
        # sample mmap-backed linear/scalar evidence without materializing full float images.
        for source,vi in ((source,vi) for source,selected in sources for vi in selected):
            interrupt()
            if not np.any(source.masks[vi].read()):
                continue
            mask_ref=store.cached(fingerprint("bake-mask-v3-preserve-pale-surfaces",source.images[vi],source.masks[vi],color),
                lambda source=source,vi=vi:valid_view_mask(source.images[vi].read(),source.masks[vi].read(),0.))
            evidence_key=fingerprint("bake-evidence-v2",source.images[vi],mask_ref,source.task,semantic)
            def evidence(source=source,vi=vi,mask_ref=mask_ref):
                raw=source.images[vi].read()
                image=np.asarray(raw,np.float32)/(255. if raw.dtype==np.uint8 else 1.)
                mask=mask_ref.read()
                if mask.shape!=image.shape[:2]:
                    raise ValueError("View image and foreground mask dimensions differ")
                if source.task=="mr" and semantic in ("roughness","metallic"):
                    channel=1 if semantic=="roughness" else 2
                    image=np.repeat(image[...,channel:channel+1],3,axis=2)
                if color:
                    image=srgb_to_linear(image)
                return image[...,:3]*mask[...,None]
            evidence_ref=store.cached(evidence_key,evidence)
            depth_ref=source.gbuffer.depths[vi]
            ramp_ref=store.cached(fingerprint("depth-edge-v2",mask_ref,depth_ref,edge_pixels),
                lambda mask_ref=mask_ref,depth_ref=depth_ref:projection_ramp(depth_ref.read(),mask_ref.read(),edge_pixels))
            # Precompute a blurred low frequency field on the per-view evidence.
            # It is cached and only used in detail_preserving mode, avoiding
            # recomputation for every UV atlas tile.
            low_ref=None
            if color and blend_mode=="detail_preserving":
                from scipy.ndimage import gaussian_filter
                def compute_low(evidence_ref=evidence_ref,mask_ref=mask_ref):
                    rgb=np.asarray(evidence_ref.read(),np.float32)
                    m=np.asarray(mask_ref.read(),np.float32)
                    # Premultiplied mask normalization to avoid white silhouettes.
                    blur_m=gaussian_filter(m, sigma=3.0)
                    blur_rgb=gaussian_filter(rgb,sigma=(3.,3.,0.))
                    return blur_rgb/np.maximum(blur_m[...,None],1e-4)
                low_ref=store.cached(fingerprint("view-low-v1",evidence_ref,mask_ref),compute_low)
            prepared[len(prepared)]=(source,vi,evidence_ref,mask_ref,ramp_ref,low_ref)
        face_buffers = {}
        if visibility_mode == "hybrid" and geometry_context is not None:
            for src,_,_,_,_,_ in prepared.values():
                key=src.plan.identity
                if key not in face_buffers:
                    face_buffers[key]=get_camera_face_ids(geometry_context,src.plan,policy)
        for tile in atlas.tiles:
            interrupt()
            p=tile.positions.read()
            n=tile.normals.read()
            h,w=p.shape[:2]
            valid=tile.face_ids.read()>=0
            points=np.asarray(p[valid],np.float32)
            normals=np.asarray(n[valid],np.float32)
            count=len(points)
            if not count:
                continue
            acc=np.zeros((count,3),np.float32)
            total=np.zeros(count,np.float32)
            best=np.zeros(count,np.float32)
            best_color=np.zeros((count,3),np.float32)
            quality=np.zeros(count,np.float32)
            detail_acc=np.zeros((count,3),np.float32)
            detail_total=np.zeros(count,np.float32)
            # Band-separated fusion intentionally prefers the strongest view
            # for edges/patterns, not low-frequency shading.
            low_acc=np.zeros((count,3),np.float32)
            high_acc=np.zeros((count,3),np.float32)
            high_total=np.zeros(count,np.float32)
            tile_reasons=np.full(count,5,np.uint8) if visibility_diagnostics else None
            rank_order=np.asarray([5,0,3,1,2,4],np.uint8) if visibility_diagnostics else None
            face_ids_atlas=np.asarray(tile.face_ids.read()[valid],np.int32)
            for view_idx,(source,vi,evidence_ref,mask_ref,ramp_ref,low_ref) in enumerate(prepared.values()):
                interrupt()
                image=evidence_ref.read()
                mask=mask_ref.read()
                # Mask-normalized interpolation prevents white-background bleeding at silhouettes.
                depth=source.gbuffer.depths[vi].read()
                mv=np.asarray(source.plan.matrices[vi],np.float32)
                camera=points@mv[:3,:3].T+mv[:3,3]
                z=-camera[:,2]
                u=.5+FOCAL*camera[:,0]/np.maximum(z,1e-8)
                v=.5-FOCAL*camera[:,1]/np.maximum(z,1e-8)
                inside=(z>.05)&(u>=0)&(u<1)&(v>=0)&(v<1)
                dh,dw=depth.shape
                visible, uncertain, depth_quality, status=subpixel_visibility(z,u,v,depth,base_tolerance=depth_tolerance)
                # Exact ray/triangle checks ONLY for ambiguous projected texels.
                # Bounded to 4096 candidates per view/tile to keep 8GB workflows practical.
                if visibility_mode == "hybrid" and geometry_context is not None and np.any(uncertain):
                    candidates=np.flatnonzero(uncertain)
                    candidates=candidates[:4096]
                    face_map=face_buffers[source.plan.identity][vi].read()
                    candidate_faces=camera_face_neighbors(face_map,u[candidates],v[candidates])
                    eye=np.linalg.inv(mv)[:3,3]
                    accepted=triangle_ray_fallback(points[candidates],eye,candidate_faces,
                                                   geometry_context.vertices,geometry_context.faces,
                                                   depth_epsilon=max(.006,depth_tolerance*2))
                    visible[candidates[accepted]]=True
                    depth_quality[candidates[accepted]]=.4
                    recovered_rays+=int(accepted.sum())
                rejected_depth+=int((inside&~visible).sum())
                if tile_reasons is not None:
                    candidate_reasons=np.full(count,5,np.uint8)
                    candidate_reasons[inside]=2
                center=np.linalg.inv(mv)[:3,3]
                direction=center-points
                direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-8)
                cosine=np.clip(np.einsum("ij,ij->i",normals,direction),0,1)
                selection=np.flatnonzero(visible&(cosine>.05))
                if not len(selection):
                    if tile_reasons is not None:
                        candidate_reasons[visible&(cosine<=.05)]=4
                        better_reason=rank_order[candidate_reasons]<rank_order[tile_reasons]
                        tile_reasons[better_reason]=candidate_reasons[better_reason]
                    continue
                ih,iw=image.shape[:2]
                px,py=u[selection]*iw-.5,v[selection]*ih-.5
                alpha=bilinear(mask,px,py)[:,0]
                edge=bilinear(store.small(ramp_ref),u[selection]*dw-.5,v[selection]*dh-.5)[:,0]
                selected_cosine=cosine[selection]
                # Bilinear mask alpha is fractional at silhouettes; requiring >.99
                # previously rejected valid subpixel pixels around the face/hood.
                weight=np.where(alpha>.55,selected_cosine**power*np.maximum(edge,.15)*depth_quality[selection],0).astype(np.float32)
                missing_mask+=int(np.count_nonzero(alpha<=.55))
                if tile_reasons is not None:
                    candidate_reasons[visible&(cosine<=.05)]=4
                    candidate_reasons[selection]=3
                    candidate_reasons[selection[weight>0]]=1
                    better_reason=rank_order[candidate_reasons]<rank_order[tile_reasons]
                    tile_reasons[better_reason]=candidate_reasons[better_reason]
                sample=bilinear(image,px,py)/np.maximum(alpha[:,None],1e-8)
                acc[selection]+=sample*weight[:,None]
                total[selection]+=weight
                detail_weight=weight*weight
                detail_acc[selection]+=sample*detail_weight[:,None]
                detail_total[selection]+=detail_weight
                if low_ref is not None:
                    low=bilinear(low_ref.read(),px,py)
                    # The high band comes from the same view and UV coordinates.
                    high=sample-low
                    low_acc[selection]+=low*weight[:,None]
                    # Stronger view preference for local detail, while weights
                    # remain continuous at geometric triangle boundaries.
                    sharp_w=weight**2
                    high_acc[selection]+=high*sharp_w[:,None]
                    high_total[selection]+=sharp_w
                better=weight>best[selection]
                best_color[selection[better]]=sample[better]
                best[selection[better]]=weight[better]
                quality[selection]=np.maximum(quality[selection],np.where(weight>0,selected_cosine*edge,0))
                del image,mask,ramp_ref
            observed=total>1e-5
            values=acc/np.maximum(total[:,None],1e-8)
            # Squared weights sharpen the preferred view continuously. A hard argmax
            # switches whole facial details at a triangle/view boundary when two
            # generated cameras disagree. Keep that old behavior only on request.
            preferred=detail_acc/np.maximum(detail_total[:,None],1e-20)
            values=(1-best_view_mix)*values+best_view_mix*preferred
            if color and blend_mode=="detail_preserving" and np.any(high_total):
                multiband=(low_acc/np.maximum(total[:,None],1e-20) +
                           high_acc/np.maximum(high_total[:,None],1e-20))
                values=(1-best_view_mix)*values+best_view_mix*multiband
            if color:
                values=linear_to_srgb(values)
            region=output[tile.y:tile.y+h,tile.x:tile.x+w]
            region[valid]=np.clip(values,0,1)
            seen[tile.y:tile.y+h,tile.x:tile.x+w][valid]=observed
            confidence[tile.y:tile.y+h,tile.x:tile.x+w][valid]=quality
            if reasons is not None:
                area=reasons[tile.y:tile.y+h,tile.x:tile.x+w]
                tile_reasons[observed]=1
                area[valid]=tile_reasons
        cover=np.asarray(atlas.uv_coverage.read())
        holes=cover&~seen
        if original is not None:
            original=np.asarray(original,np.float32)
            if original.shape != output.shape:
                raise ValueError("Existing atlas must have the same dimensions and 3 channels")
            output[holes]=original[holes]
        # A texture exported directly from this node must not bilinearly sample black
        # outside UV islands. This is gutter extension only, never hidden hole filling.
        padded=pad_gutters(output,cover,seen|(holes if original is not None else False),gutter)
        report.update(observed_uv_fraction=float(seen.sum()/max(cover.sum(),1)),
                      preserved_uv_fraction=float(holes.sum()/max(cover.sum(),1)) if original is not None else 0.,
                      coverage_metric="fraction of occupied atlas texels, not mesh surface area",
                      confidence_kind="geometric view quality; not model uncertainty",
                      blend_mode=blend_mode,source_views=sum(len(indices) for _,indices in sources),
                      visibility_mode=visibility_mode,ray_candidates_recovered=recovered_rays,
                      rejected_depth_candidates=rejected_depth,rejected_mask_candidates=missing_mask,
                      gutter_pixels=gutter,gutter_texels=padded,unobserved_texels=int(holes.sum()))
        if reasons is not None:
            report['visibility_preview']=diagnostic_preview(reasons,cover)
    info=MapInfo(atlas.atlas_id,semantic,t,t,provenance=views.job_id)
    return output,info,seen.astype(np.float32),confidence,holes.astype(np.float32),report


def pad_gutters(image,coverage,known,padding):
    """Extend valid colors into gutters in place; preserve every occupied UV texel."""
    if padding<=0 or not np.any(known):
        return 0
    from scipy.ndimage import distance_transform_edt
    distance,indices=distance_transform_edt(~known,return_indices=True)
    use=(~coverage)&(distance<=padding)
    image[use]=image[indices[0][use],indices[1][use]]
    return int(use.sum())


def repair_atlas(image,holes,atlas,mode="none",padding=8,max_fill_distance=32):
    from scipy.ndimage import distance_transform_edt, find_objects
    out=np.array(image,dtype=np.float32,copy=True)
    charts=np.asarray(atlas.chart_ids.read())
    if out.shape[:2]!=charts.shape or holes.shape!=charts.shape:
        raise ValueError("Repair image, holes, and atlas must have matching dimensions")
    repaired=np.zeros(charts.shape,bool)
    if mode=="nearest_in_chart":
        # Process each chart's bounding box; never use nearest neighbours across different islands.
        for chart,region in enumerate(find_objects(charts),1):
            if region is None:
                continue
            interrupt()
            label=charts[region]==chart
            missing=(holes[region]>.5)&label
            known=label&~(holes[region]>.5)
            if not missing.any() or not known.any():
                continue
            distance,indices=distance_transform_edt(~known,return_indices=True)
            use=missing&(distance<=max_fill_distance)
            local=out[region]
            local[use]=local[indices[0][use],indices[1][use]]
            repaired[region][use]=True
    elif mode in ("nearest_surface", "normal_aware_surface"):
        from scipy.spatial import cKDTree
        # Tile arrays hold the exact surface position. Neighbours are chosen in 3D
        # instead of copying long strips across an arbitrary 2D UV chart.
        known=(charts>0)&~(holes>.5)
        missing=(charts>0)&(holes>.5)
        if known.any() and missing.any():
            ram_preflight(int(known.sum())*96+image.nbytes,256)
            positions=np.empty((int(known.sum()),3),np.float32)
            colors=np.empty_like(positions)
            norm_known=np.empty_like(positions)
            cursor=0
            for tile in atlas.tiles:
                p=tile.positions.read();h,w=p.shape[:2]
                region=np.s_[tile.y:tile.y+h,tile.x:tile.x+w]
                select=known[region];count=int(select.sum())
                positions[cursor:cursor+count]=p[select]
                colors[cursor:cursor+count]=out[region][select]
                if mode=="normal_aware_surface":
                    norm_known[cursor:cursor+count]=tile.normals.read()[select]
                cursor+=count
            tree=cKDTree(positions)
            for tile in atlas.tiles:
                interrupt()
                p=tile.positions.read();h,w=p.shape[:2]
                region=np.s_[tile.y:tile.y+h,tile.x:tile.x+w]
                select=missing[region]
                if not select.any():continue
                # Larger search pool allows excluding a near-but-opposite surface.
                k=min(12 if mode=="normal_aware_surface" else 4,len(positions))
                distances,ids=tree.query(p[select],k=k)
                if k==1:
                    result=colors[ids]
                elif mode=="normal_aware_surface":
                    normals=tile.normals.read()[select]
                    cos=np.einsum("ijk,ik->ij",norm_known[ids],normals)
                    # Reject opposing and sharply oblique surface directions.
                    good=cos>0.5
                    weights=np.where(good,1./np.maximum(distances,1e-4)**2,0.)
                    # Never copy an unrelated neighbor when no matching normal
                    # exists: fall back to the closest texel, with low confidence.
                    empty=weights.sum(axis=1)<1e-15
                    if empty.any(): weights[empty,0]=1.
                    result=(colors[ids]*weights[...,None]).sum(axis=1)/weights.sum(axis=1)[:,None]
                else:
                    result=colors[ids].mean(axis=1)
                out[region][select]=result
                repaired[region][select]=True
    elif mode!="none":
        raise ValueError("Unknown repair mode")
    if padding>0:
        known=(charts>0)&(~(holes>.5)|repaired)
        if known.any():
            distance,indices=distance_transform_edt(~known,return_indices=True)
            use=(charts==0)&(distance<=padding)
            out[use]=out[indices[0][use],indices[1][use]]
            repaired[use]=True
    return out,repaired.astype(np.float32),{"stage":"repair_atlas","mode":mode,"changed_pixels":int(repaired.sum())}
