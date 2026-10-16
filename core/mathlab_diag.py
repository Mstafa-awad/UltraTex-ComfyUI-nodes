"""CPU-only UltraTex LoRA diagnostics. Importable without ComfyUI.

Reports 4B structural coverage, projection-coordinate numerical fidelity,
retention of 9B source update energy, and optionally image comparisons.
NONE of these certifies 9B-to-4B feature/behavior transfer.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import re
import torch
from safetensors import safe_open

try:
    from .mathlab_converter import (parse_model, shapes, project, depth_weights, blend_svd,
        lowrank_norm, PARTS, DST_N, DST_H, SRC_H, select_energy_groups)
except ImportError:
    from .mathlab_converter import (parse_model, shapes, project, depth_weights, blend_svd,
        lowrank_norm, PARTS, DST_N, DST_H, SRC_H, select_energy_groups)

def inner_delta(a,b,c,d):
    # Frobenius inner product (B@A) : (D@C) with O(rank^2 * channels) memory.
    return float(((b.T@d)*(a@c.T)).sum())

def inspect_4b(source_file, target_file, deep=False):
    """No model loading, no GPU. Names and dimensions are checked regardless of deep."""
    results={'source':str(source_file),'target':str(target_file),
             'warning':'Numerical matching in a projected coordinate space is NOT semantic equivalence between Klein 9B and Klein 4B.',
             'architecture_equivalence_percent':None,'behavior_equivalence_percent':None}
    with safe_open(str(source_file),framework='pt',device='cpu') as src, safe_open(str(target_file),framework='pt',device='cpu') as dst:
        rank,paths=parse_model(src)
        meta=dst.metadata() or {}
        groups_txt=meta.get('selected_source_groups','')
        groups=[int(x) for x in groups_txt.split(',')] if groups_txt else None
        results['metadata']=meta
        results['rank']=rank
        actual=set(dst.keys()); expected=set()
        for kind,parts in PARTS.items():
            for i in range(DST_N[kind]):
                for part in parts:
                    key=f'diffusion_model.{kind}.{i}.{part}'
                    ak=key+'.lora_down.weight';bk=key+'.lora_up.weight'
                    expected.update((ak,bk))
                    if ak in actual and bk in actual:
                        aa=dst.get_slice(ak).get_shape();bb=dst.get_slice(bk).get_shape()
                        if tuple(aa)!=shapes(kind,part,DST_H,rank)[0] or tuple(bb)!=shapes(kind,part,DST_H,rank)[1]:
                            raise ValueError(f'Wrong 4B tensor shapes: {key}: A{aa}, B{bb}')
        if expected!=actual:raise ValueError(f'Incomplete 4B adapter: missing {list(expected-actual)[:5]}, extra {list(actual-expected)[:5]}')
        results['structural_coverage_percent']=100.
        results['mapped_modules']=len(expected)//2
        results['mapped_tensors']=len(expected)
        results['original_modules']=len(paths)
        results['semantic_similarity']='UNKNOWN; requires 9B and 4B model inference'
        if not deep:
            results['numerical_test']='Run deep=True to compare projected source LoRA updates'
            return results
        if groups is None or len(groups)!=24 or len(set(groups))!=24 or any(not 0<=x<32 for x in groups):
            raise ValueError('Deep numerical comparison requires a converter output with selected_source_groups metadata')
        depth=meta.get('depth_mapping','nearest')
        scale=float(meta.get('source_scale','1.0'))
        total_full=0.;total_proj=0.;total_conv=0.;total_dot=0.
        layer_stats=[]
        for kind,parts in PARTS.items():
            for i in range(DST_N[kind]):
                picks=depth_weights(kind,i,depth)
                for part in parts:
                    p_factors=[];factors=[]
                    for ix,w in picks:
                        keys=paths[(kind,ix,part)]
                        a=src.get_tensor(keys['A']).float();b=src.get_tensor(keys['B']).float()
                        pa,pb=project(a,b,groups)
                        p_factors.append((pa,pb,w))
                        factors.append((a,b,w))
                    # Correct reference is the mixture of target-subspace source updates,
                    # not the original unprojectable full 9B update.
                    ar=torch.cat([math.sqrt(w)*a for a,b,w in p_factors],dim=0)
                    br=torch.cat([math.sqrt(w)*b for a,b,w in p_factors],dim=1)
                    af=torch.cat([math.sqrt(w)*a for a,b,w in factors],dim=0)
                    bf=torch.cat([math.sqrt(w)*b for a,b,w in factors],dim=1)
                    prefix=f'diffusion_model.{kind}.{i}.{part}'
                    at=dst.get_tensor(prefix+'.lora_down.weight').float()
                    bt=dst.get_tensor(prefix+'.lora_up.weight').float()
                    full=lowrank_norm(af,bf)
                    proj=lowrank_norm(ar,br)
                    conv=lowrank_norm(at,bt)
                    dot=inner_delta(ar,br,at,bt)
                    cosine=dot/max(proj*conv,1e-20)
                    retain=(proj/full)**2 if full>0 else 0
                    layer_stats.append({'layer':prefix,'source_coordinate_retention_pct':round(100*retain,3),
                        'projected_subspace_cosine_pct':round(100*cosine,3),
                        'source_full_update_norm':round(full,4), 'projected_update_norm':round(proj,4),
                        'converted_update_norm':round(conv,4)})
                    total_full+=full*full;total_proj+=proj*proj;total_conv+=conv*conv;total_dot+=dot
        results['nine_b_source_update_energy_retained_pct']=round(100*total_proj/max(total_full,1e-20),2)
        results['projected_source_to_converted_cosine_pct']=round(100*total_dot/math.sqrt(max(total_proj*total_conv,1e-20)),2)
        results['numerical_test']='Sources are 9B weights projected into 4B-SHAPED source subspace; NOT actual 4B backbone activations'
        results['per_layer']=layer_stats
    return results

def compare_images(a,b):
    """Aligned tensor images, RGB floats [B,H,W,C]. Uses background-masked measures."""
    a=torch.as_tensor(a).detach().float().cpu().clamp(0,1)
    b=torch.as_tensor(b).detach().float().cpu().clamp(0,1)
    if a.ndim!=4 or b.ndim!=4 or a.shape[-1]<3 or b.shape[-1]<3:
        raise ValueError('Both image connections must be ComfyUI IMAGE [batch,height,width,channels]')
    if a.shape[:3]!=b.shape[:3]:
        raise ValueError(f'9B and 4B views must be EXACTLY aligned with equal batch and size, got {a.shape}, {b.shape}')
    a=a[...,:3]; b=b[...,:3]
    # Foreground excludes pixels where BOTH images are near-white.
    fg=((a<.97).any(-1)|(b<.97).any(-1))
    if int(fg.sum())<32:raise ValueError('Images have insufficient nonwhite foreground for comparison')
    d=(a-b).abs()
    picked=d[fg]
    mae=float(picked.mean());rmse=float((picked.square().mean()).sqrt())
    # Fraction of foreground pixels for which ALL channels agree to 2/255.
    tolerance=2/255
    pixel_agree=float(((d.amax(dim=-1)<=tolerance)&fg).sum()/fg.sum())
    psnr=-20*math.log10(max(rmse,1e-10))
    va=a[fg].reshape(-1);vb=b[fg].reshape(-1)
    cc=float(((va-va.mean())*(vb-vb.mean())).mean()/(va.std(unbiased=False)*vb.std(unbiased=False)+1e-10))
    ea=((a[:,1:]-a[:,:-1]).abs().mean(-1))
    eb=((b[:,1:]-b[:,:-1]).abs().mean(-1))
    mask=fg[:,1:]|fg[:,:-1]
    ed_a=ea[mask];ed_b=eb[mask]
    edge=float(((ed_a-ed_a.mean())*(ed_b-ed_b.mean())).mean()/(ed_a.std(unbiased=False)*ed_b.std(unbiased=False)+1e-10))
    preview=torch.cat((a,b,(d*3).clamp(0,1)),dim=2)
    return {'foreground_mae':round(mae,6),'foreground_psnr_db':round(psnr,2),
        'foreground_exact_rgb_pixel_agreement_percent':round(100*pixel_agree,3),
        'foreground_color_correlation':round(cc,5),'vertical_edge_magnitude_correlation':round(edge,5),
        'pixel_exact_100_percent':bool(pixel_agree==1),
        'warning':'These metrics compare rendered pixels. 100% pixels only indicates identical outputs for these test images, not equivalent model internals.'},preview


def compare_lora_effects(img9_on,img9_off,img4_on,img4_off):
    """Compare changes caused by each adapter, rather than confusing base model differences.

    Requires four same-shape image batches. Approximate, non-linear image response.
    """
    images=[torch.as_tensor(x).detach().float().cpu().clamp(0,1)[...,:3] for x in
            (img9_on,img9_off,img4_on,img4_off)]
    if any(x.ndim!=4 for x in images) or len({tuple(x.shape) for x in images})!=1:
        raise ValueError('All four on/off image batches need matching [B,H,W,3] dimensions and aligned views')
    on9,off9,on4,off4=images
    m=(((on9<.97).any(-1)|(off9<.97).any(-1)|
        (on4<.97).any(-1)|(off4<.97).any(-1)))
    if int(m.sum())<32:raise ValueError('Insufficient foreground')
    delta9=(on9-off9)[m].reshape(-1)
    delta4=(on4-off4)[m].reshape(-1)
    n9=float(torch.linalg.vector_norm(delta9)); n4=float(torch.linalg.vector_norm(delta4))
    cosine=float((delta9@delta4)/max(n9*n4,1e-16))
    return {'nine_b_lora_image_effect_norm':round(n9,4),
            'four_b_lora_image_effect_norm':round(n4,4),
            'effect_magnitude_ratio_4b_over_9b':round(n4/max(n9,1e-10),4),
            'adapter_effect_direction_cosine':round(cosine,6),
            'warning':'Image-difference alignment is more diagnostic than comparing base-model images but still cannot certify LoRA equivalence.'}

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser(description='Report source-coordinate math fidelity; no 4B model needed')
    a.add_argument('original_9b');a.add_argument('converted_4b');a.add_argument('--deep',action='store_true')
    args=a.parse_args()
    print(json.dumps(inspect_4b(args.original_9b,args.converted_4b,args.deep),indent=2))
