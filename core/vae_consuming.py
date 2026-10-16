"""Inference-only VAE residual scheduling. No change to arithmetic operators.
Projected shortcut first permits consuming the oversized input in-place.
Native aten GroupNorm computes moments before its aliased output write.
Research module: scope restores every forward; never patches global Torch.
"""
from contextlib import contextmanager
import torch

def consume_norm(norm,x):
 import comfy.ops
 with comfy.ops.CastBiasWeightContext(norm,x,offloadable=True) as (weight,bias):
  stats_dtype=weight.dtype if weight is not None and weight.dtype!=x.dtype else x.dtype
  mean=torch.empty((x.shape[0],norm.num_groups),device=x.device,dtype=stats_dtype);rstd=torch.empty_like(mean)
  torch.ops.aten.native_group_norm.out(x,weight,bias,x.shape[0],x.shape[1],x.shape[2]*x.shape[3],norm.num_groups,norm.eps,out0=x,out1=mean,out2=rstd)
 return x
@contextmanager
def consume_residuals(model):
 changes=[];counts=dict(projected_consumptions=0,norm2_consumptions=0,residual_adds=0)
 for block in model.modules():
  if type(block).__name__!='ResnetBlock' or not all(hasattr(block,n) for n in ('norm1','norm2','conv1','conv2','swish','dropout')):continue
  if not isinstance(block.conv1,torch.nn.Conv2d) or not isinstance(block.norm1,torch.nn.GroupNorm):continue
  original=block.forward
  def forward(x,temb=None,conv_carry_in=None,conv_carry_out=None,block=block,original=original):
   if torch.is_grad_enabled() or x.ndim!=4 or not x.is_contiguous() or conv_carry_in is not None or conv_carry_out is not None or temb is not None:
    return original(x,temb,conv_carry_in=conv_carry_in,conv_carry_out=conv_carry_out)
   if block.in_channels!=block.out_channels:
    residual=(block.conv_shortcut(x) if block.use_conv_shortcut else block.nin_shortcut(x))
    h=consume_norm(block.norm1,x);counts['projected_consumptions']+=1
   else:residual=x;h=block.norm1(x)
   h=block.conv1(block.swish(h))
   h=consume_norm(block.norm2,h);counts['norm2_consumptions']+=1
   h=block.conv2(block.dropout(block.swish(h)))
   h.add_(residual);counts['residual_adds']+=1
   return h
  changes.append((block,'forward' in block.__dict__,block.__dict__.get('forward')));block.forward=forward
 try:yield counts
 finally:
  for block,had,value in changes:
   if had:block.forward=value
   else:delattr(block,'forward')
