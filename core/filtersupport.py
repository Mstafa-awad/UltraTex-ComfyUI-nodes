"""Exact source-pixel support of bilinear mips and bounded cubic reconstruction.

The UV bounds describe the samples actually read, not a Gaussian-radius proxy.
This prevents a coarse mip from reaching a surface outside the edge query.
"""


def edge_overlap(view, u, v, footprint, torch):
    """Return queries whose full colour support may cross a geometric edge."""
    # Bound temporary integer/float arrays independently of the projection
    # chunk. The result is identical for every query and needs one byte each.
    chunk = 32768
    if u.numel() > chunk:
        return torch.cat([_edge_overlap(view, u[i:i+chunk], v[i:i+chunk],
                                       footprint[i:i+chunk], torch)
                          for i in range(0, u.numel(), chunk)])
    return _edge_overlap(view, u, v, footprint, torch)


def _edge_overlap(view, u, v, footprint, torch):
    size=view.S;depth_size=view.R
    lod=torch.log2(footprint.clamp_min(1)).clamp_max(len(view.levels)-1)
    upper=torch.where(lod>.05,torch.ceil(lod),torch.zeros_like(lod))
    scale=torch.pow(2.,upper)
    # Bilinear sampling of an averaged mip reads two adjacent source blocks.
    bx=torch.floor(u*size/scale-.5);by=torch.floor(v*size/scale-.5)
    left=bx*scale;right=(bx+2)*scale-1
    top=by*scale;bottom=(by+2)*scale-1
    # Bounded cubic can additionally read this 4x4 source window. Including it
    # for all queries is conservative; larger mip supports already contain it.
    cx=torch.floor(u*size-.5);cy=torch.floor(v*size-.5)
    left=torch.minimum(left,cx-1).clamp(0,size-1)
    right=torch.maximum(right,cx+2).clamp(0,size-1)
    top=torch.minimum(top,cy-1).clamp(0,size-1)
    bottom=torch.maximum(bottom,cy+2).clamp(0,size-1)
    lo_u=(left+.5)/size;hi_u=(right+.5)/size
    lo_v=(top+.5)/size;hi_v=(bottom+.5)/size
    tr=view.transform
    if tr is not None:
        lo_u=((lo_u*size-tr.pad_left)/tr.resized_width*tr.width+tr.left)/tr.original_width
        hi_u=((hi_u*size-tr.pad_left)/tr.resized_width*tr.width+tr.left)/tr.original_width
        lo_v=((lo_v*size-tr.pad_top)/tr.resized_height*tr.height+tr.top)/tr.original_height
        hi_v=((hi_v*size-tr.pad_top)/tr.resized_height*tr.height+tr.top)/tr.original_height
    # Include both possible nearest depth pixels at a half-pixel tie.
    x0=torch.floor(lo_u*depth_size-.5).long().clamp(0,depth_size-1)
    x1=torch.ceil(hi_u*depth_size-.5).long().clamp(0,depth_size-1)
    y0=torch.floor(lo_v*depth_size-.5).long().clamp(0,depth_size-1)
    y1=torch.ceil(hi_v*depth_size-.5).long().clamp(0,depth_size-1)
    span=torch.maximum(x1-x0+1,y1-y0+1)
    level=torch.ceil(torch.log2(span.float())).long().clamp(0,len(view.surface_edges)-1)
    # With odd-sized buffers the edge pyramid can stop early. Never declare
    # such a window safe just by looking at its four far-apart corner cells.
    hit=span>torch.pow(2,level)
    for k,edges in enumerate(view.surface_edges):
        ids=((level==k)&~hit).nonzero(as_tuple=True)[0]
        if not ids.numel():continue
        width=edges.shape[1];height=edges.shape[0]
        for x,y in [(x0,y0),(x0,y1),(x1,y0),(x1,y1)]:
            xx=(x[ids]>>k).clamp(0,width-1);yy=(y[ids]>>k).clamp(0,height-1)
            hit[ids]|=edges[yy,xx]
    return hit
