"""Explicit TEST allocation cap after ComfyUI initializes Torch, before graphs."""
import os

def apply_native_cap():
    cap_mb=int(os.environ.get('UTX_FULL_CONTEXT_STARTUP_CAP_MB','0'))
    if cap_mb<=0:return None
    import torch
    if not torch.cuda.is_available():return None
    device=torch.cuda.current_device();total=torch.cuda.get_device_properties(device).total_memory
    previous=torch.cuda.get_per_process_memory_fraction(device)
    cap=min(1.,cap_mb*2**20/total)
    if 0.<previous<=1.:cap=min(cap,previous)
    torch.cuda.set_per_process_memory_fraction(float(cap),device)
    print(f'[UltraTex TEST] Explicit native allocator cap at registration: {cap*total/2**20:.0f} MiB; restart without UTX_FULL_CONTEXT_STARTUP_CAP_MB for rollback',flush=True)
    return cap
