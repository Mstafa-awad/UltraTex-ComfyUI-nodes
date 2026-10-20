#!/usr/bin/env python3
"""UltraTex Klein 9B -> 4B: experimental, LoRA-informed projection and SVD blending.

Only source LoRA weights are used. No alignment to 4B backbone can be inferred
from these weights. Shape correctness and in-source subspace preservation DO NOT
mean semantic equivalence to an actual FLUX.2 Klein 9B model.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_H, DST_H, GROUP, SRC_N, DST_N, RANK = 4096, 3072, 128, {"double_blocks":8,"single_blocks":24},{"double_blocks":5,"single_blocks":20},64
PARTS={"double_blocks":("img_attn.proj","img_attn.qkv","txt_attn.proj","txt_attn.qkv"),"single_blocks":("linear1","linear2")}
RX=re.compile(r"^(?:base_model\.model\.|diffusion_model\.|model\.)?(double_blocks|single_blocks)\.(\d+)\.(img_attn\.proj|img_attn\.qkv|txt_attn\.proj|txt_attn\.qkv|linear1|linear2)\.lora_([AB])(?:\.default)?\.weight$")

def shapes(kind,part,hidden,rank):
    if kind=="double_blocks":
        in_mult, out_mult=1,(3 if part.endswith('qkv') else 1)
    elif part=='linear1':in_mult,out_mult=1,9
    elif part=='linear2':in_mult,out_mult=4,1
    else:raise ValueError((kind,part))
    return (rank,hidden*in_mult),(hidden*out_mult,rank)

def parse_model(source):
    paths={}
    for key in source.keys():
        m=RX.fullmatch(key)
        if not m:raise ValueError(f"Unsupported source key: {key}")
        kind,ix,part,ab=m.groups(); ix=int(ix)
        if kind not in PARTS or part not in PARTS[kind]:raise ValueError(f"Unsupported layer {key}")
        if ix>=SRC_N[kind]:raise ValueError(f"Unexpected 9B index {key}")
        loc=(kind,ix,part)
        if ab in paths.setdefault(loc,{}):raise ValueError(f"Duplicate {key}")
        paths[loc][ab]=key
    expected={(k,i,p) for k,ps in PARTS.items() for i in range(SRC_N[k]) for p in ps}
    if set(paths)!=expected or any(set(x)!={"A","B"} for x in paths.values()):
        raise ValueError(f"9B LoRA incomplete: missing {len(expected-set(paths))}, extra {len(set(paths)-expected)}")
    rank=source.get_slice(paths[('double_blocks',0,'img_attn.proj')]['A']).get_shape()[0]
    for (kind,ix,part),pairs in paths.items():
        if tuple(source.get_slice(pairs['A']).get_shape())!=shapes(kind,part,SRC_H,rank)[0] or tuple(source.get_slice(pairs['B']).get_shape())!=shapes(kind,part,SRC_H,rank)[1]:
            raise ValueError(f"Wrong source shape: {kind}.{ix}.{part}")
    return rank,paths

def select_energy_groups(source,paths):
    """One source-to-target coordinate mask across EVERY layer. 128-channel groups.
    Scores use actual B@A row/column energies (gauge-invariant) and
    are normalized per module / side to avoid oversized layers dominating.
    No inference about 4B coordinate alignment is implied.
    """
    values=torch.zeros((SRC_H//GROUP,),dtype=torch.float64)
    for (kind,ix,part),pair in paths.items():
        a=source.get_tensor(pair['A']).float()
        b=source.get_tensor(pair['B']).float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError(f"Non-finite: {pair}")
        # Gauge-invariant row/column energy of the ACTUAL update D=B@A:
        # diag(D^T D) = columns of A weighted by B^T B
        # diag(D D^T) = rows of B weighted by A A^T
        # This is invariant under A->T@A, B->B@T^-1, unlike raw factor norms.
        gb=b.T@b
        ga=a@a.T
        column_energy=((a.T@gb)*a.T).sum(dim=1).clamp_min(0)
        row_energy=((b@ga)*b).sum(dim=1).clamp_min(0)
        for channel_energy in (column_energy,row_energy):
            energy=channel_energy.reshape(-1,32,GROUP).sum((0,2)).double()
            values+=energy/energy.sum().clamp_min(1e-30)
        del a,b
    chosen=sorted(sorted(range(32),key=lambda i:(-values[i].item(),i))[:24])
    return chosen,values

def index_for(groups,multiple):
    base=torch.cat([torch.arange(g*GROUP,(g+1)*GROUP) for g in groups])
    return torch.cat([base+j*SRC_H for j in range(multiple)])

def project(a,b,groups):
    if a.shape[1]%SRC_H or b.shape[0]%SRC_H:raise ValueError('Bad projection shape')
    ai=index_for(groups,a.shape[1]//SRC_H)
    bi=index_for(groups,b.shape[0]//SRC_H)
    return a.index_select(1,ai).contiguous(),b.index_select(0,bi).contiguous()

def lowrank_norm(a,b):
    # Rank-small Gram norm: ||B@A||_F, avoids large dense matrix.
    return float(torch.clamp(((b.T@b)*(a@a.T).T).sum(),min=0).sqrt())

def blend_svd(factors,rank):
    """Frobenius-optimal truncated SVD of sum_i weight_i B_i A_i.
    QR on factors + tiny SVD: no dense (up to 27k x 16k) update.
    """
    if len(factors)==1:
        a,b,w=factors[0]
        return a.contiguous(),(b*w).contiguous(),0.
    bs=torch.cat([math.sqrt(w)*b for a,b,w in factors],dim=1)
    ats=torch.cat([math.sqrt(w)*a.T for a,b,w in factors],dim=1)
    qb,rb=torch.linalg.qr(bs,mode='reduced')
    qa,ra=torch.linalg.qr(ats,mode='reduced')
    u,s,vh=torch.linalg.svd(rb@ra.T,full_matrices=False)
    n=min(rank,s.numel())
    root=s[:n].clamp_min(0).sqrt()
    newb=(qb@u[:,:n])*root[None,:]
    newa=(root[:,None]*vh[:n,:])@qa.T
    tail=float(s[n:].square().sum()/s.square().sum().clamp_min(1e-30))
    return newa.contiguous(),newb.contiguous(),tail

def depth_weights(kind,i,mode):
    point=i*(SRC_N[kind]-1)/(DST_N[kind]-1)
    if mode=='nearest':
        j=min(SRC_N[kind]-1,int(math.floor(point+.5)))
        return [(j,1.0)]
    left=int(math.floor(point)); right=int(math.ceil(point)); mix=point-left
    return [(left,1-mix),(right,mix)] if right!=left else [(left,1.)]

def convert(path,out_path,*,depth='nearest',dtype='fp16',source_scale=1.,threads=0,report_path=None):
    torch.set_num_threads(max(1,threads or os.cpu_count() or 1))
    if path.resolve()==out_path.resolve():raise ValueError('Cannot overwrite source')
    if source_scale<=0:raise ValueError('Source alpha/rank must be positive')
    begin=time.perf_counter();saved={};records=[]
    dt={'fp16':torch.float16,'bf16':torch.bfloat16,'fp32':torch.float32}[dtype]
    with safe_open(str(path),framework='pt',device='cpu') as sf:
        rank,paths=parse_model(sf)
        groups,scores=select_energy_groups(sf,paths)
        print(f'[mathlab] Original: {len(paths)} 9B modules, rank {rank}; selected 24/32 coordinate groups: {groups}',flush=True)
        for kind,parts in PARTS.items():
            for dest_i in range(DST_N[kind]):
                picks=depth_weights(kind,dest_i,depth)
                for part in parts:
                    source_factors=[];sources=[];ref_energy=0.
                    for ix,w in picks:
                        keys=paths[(kind,ix,part)]
                        a=sf.get_tensor(keys['A']).float(); b=sf.get_tensor(keys['B']).float()
                        if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError(f'Non-finite: {keys}')
                        na,nb=project(a,b,groups)
                        source_factors.append((na,nb,w))
                        sources.append({'index':ix,'weight':round(w,8),'source_update_norm':round(lowrank_norm(a,b),6), 'projected_update_norm':round(lowrank_norm(na,nb),6)})
                        ref_energy+=w*(lowrank_norm(a,b)**2)
                    na,nb,trunc_loss=blend_svd(source_factors,rank)
                    # Deliberately NO renormalizing-to-original-size: would amplify unaligned directions.
                    nb=nb*source_scale
                    target_a,target_b=shapes(kind,part,DST_H,rank)
                    if tuple(na.shape)!=target_a or tuple(nb.shape)!=target_b:
                        raise AssertionError(f'Shape mismatch at {kind}.{dest_i}.{part}')
                    base=f'diffusion_model.{kind}.{dest_i}.{part}'
                    saved[base+'.lora_down.weight']=na.to(dt).contiguous()
                    saved[base+'.lora_up.weight']=nb.to(dt).contiguous()
                    records.append({'target':base,'sources':sources,'converted_update_norm':round(lowrank_norm(na,nb),6),'relative_update_energy_proxy':round(lowrank_norm(na,nb)**2/max(ref_energy,1e-30),6),'blend_SVD_discarded_energy_ratio':round(trunc_loss,8)})
                print(f'[mathlab] {kind}: {dest_i+1}/{DST_N[kind]} ({picks})',flush=True)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    meta={'architecture':'FLUX.2 Klein 4B','algorithm':'Gauge-invariant delta-energy global 24/32 group selection + low-rank optimal depth blending',
          'source_arch':'FLUX.2 Klein 9B','semantic_transfer_verified':'false','warning':'Without 4B/9B base-model alignment, this is an unverified experiment.',
          'depth_mapping':depth,'source_scale':str(source_scale),'selected_source_groups':','.join(map(str,groups))}
    save_file(saved,str(out_path),metadata=meta)
    report={'algorithm':meta['algorithm'],'warning':meta['warning'],'source':str(path),'output':str(out_path),'rank':rank,
            'selected_groups':groups,'group_energy_scores':[round(x,6) for x in scores.tolist()],
            'source_model_compatibility_percent':None,'model_behavior_equivalence_percent':None,
            'structural_layer_count':len(records),'structural_tensor_count':len(saved),
            'mean_relative_projected_energy_proxy':round(sum(x['relative_update_energy_proxy'] for x in records)/len(records),6),
            'note':'The energy proxy measures only LoRA updates in the source coordinate system. It does not test target 4B feature alignment.',
            'layers':records,'elapsed_seconds':round(time.perf_counter()-begin,2)}
    if report_path is None:report_path=out_path.with_suffix('.report.json')
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'[mathlab] Saved {out_path} ({out_path.stat().st_size/1048576:.1f} MiB)')
    print(f'[mathlab] Mean retained/blended update energy proxy: {100*report["mean_relative_projected_energy_proxy"]:.1f}% (NOT 9B behavior equivalence!)')
    print(f'[mathlab] Report {report_path}')
    return report

def main():
    ap=argparse.ArgumentParser(description='Experimental Klein 9B UltraTex LoRA -> 4B, no training')
    ap.add_argument('input',type=Path,help='Drag-and-drop original 9B .safetensors onto Convert_DROP_LORA_HERE.bat')
    ap.add_argument('--output',type=Path,default=None)
    ap.add_argument('--depth',choices=['blend','nearest'],default='nearest')
    ap.add_argument('--dtype',choices=['fp16','bf16','fp32'],default='fp16')
    ap.add_argument('--source-scale',type=float,default=1.0,help='PEFT alpha/rank: if source alpha is 64 and rank is 64, use 1')
    ap.add_argument('--threads',type=int,default=0,help='CPU math threads; 0 uses all logical processors')
    args=ap.parse_args()
    if not args.input.is_file():ap.error('Input file not found')
    if args.input.suffix.lower()!='.safetensors':ap.error('Please drag extracted .safetensors, not RAR/ZIP')
    output=args.output or args.input.with_name(args.input.stem+('_Klein4B_ENERGY_'+args.depth.upper()+'.safetensors'))
    try:convert(args.input,output,depth=args.depth,dtype=args.dtype,source_scale=args.source_scale,threads=args.threads)
    except Exception as e:
        print(f'[mathlab] FAILED: {type(e).__name__}: {e}',file=sys.stderr)
        raise
if __name__=='__main__':main()
