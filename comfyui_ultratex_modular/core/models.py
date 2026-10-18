from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch

from .contracts import ModelInfo
from .runtime import object_identity


@lru_cache(maxsize=32)
def _hash_file(path, size, mtime_ns):
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        while block:=stream.read(8*2**20):
            digest.update(block)
    return digest.hexdigest()


def file_hash(path):
    p=Path(path).resolve()
    stat=p.stat()
    return _hash_file(str(p),stat.st_size,stat.st_mtime_ns)


_KNOWN_ADAPTER_TASKS={
    # Verified against ModelScope's Yibo-Zhang/UltraTex repo/files API, 2026-10-10.
    # flux2_mr/lora/dit_lora.safetensors and flux2/lora/dit_lora.safetensors respectively.
    "13e94f5a30a789b1e3e76d8c96cbba5f52fb2f68411e56377f382e145da23fc1":"mr",
    "3b1915177a81bc2ab43f501355bf39113df0128e6cbc6fedf8b8621d5e06a01c":"albedo",
}


def validate_adapter_task(path,task,digest=None):
    """Known artifact hashes prevent renamed MR weights from masquerading as albedo."""
    if task not in ("albedo","mr"):
        raise ValueError("Task must be albedo or mr")
    digest=digest or file_hash(path)
    expected=_KNOWN_ADAPTER_TASKS.get(digest)
    if expected is None:
        import re
        if re.search(r"(?:^|[_ .-])(?:mr|metallic.?roughness)(?:[_ .-]|$)",Path(path).stem.lower()):
            expected="mr"
        elif "albedo" in Path(path).stem.lower():
            expected="albedo"
    if expected is not None and task!=expected:
        raise ValueError(f"Adapter/task mismatch: {Path(path).name} is the {expected} adapter, "
                         f"but task={task}. Select the weights for {task}, or correct the task. "
                         "Known file hashes take priority over filenames, which may be swapped. "
                         "Published albedo SHA-256 starts 3b191517; MR starts 13e94f5a.")
    return digest,expected


def select_adapter_name(names,task,resolve_path):
    """Choose a verified installed UltraTex artifact when generating example graphs.

    Limit IO to explicitly named UltraTex files; do not hash a user's whole LoRA library.
    Normal node execution still accepts arbitrary names through validate_adapter_task.
    """
    fallback="ultratex_flux2.safetensors" if task=="albedo" else "ultratex_flux2_mr.safetensors"
    candidates=[n for n in names if "ultratex" in Path(n).name.lower() and n.lower().endswith(".safetensors")]
    for name in sorted(candidates,key=lambda n:n!=fallback):
        if _KNOWN_ADAPTER_TASKS.get(file_hash(resolve_path(name)))==task:
            return name
    return fallback


def validate_backbone(model, require_gguf=False):
    import comfy.model_base
    if not isinstance(model.model,comfy.model_base.Flux2):
        raise TypeError("Load FLUX.2 Klein base 9B with a native ComfyUI or GGUF diffusion-model loader")
    dm=model.model.diffusion_model
    if dm.hidden_size!=4096 or dm.patch_size!=1 or len(dm.params.axes_dim)!=4:
        raise ValueError("Backbone dimensions do not match the released Klein 9B UltraTex adapters")
    if type(model.model.latent_format).__name__!="Flux2":
        raise ValueError("Unsupported latent format; expected native Flux2 (128 channels, identity conversion)")
    if require_gguf and not any("GGUF" in cls.__name__.upper() for cls in type(model).__mro__):
        raise ValueError("This package's generation path requires the GGUF model patcher. Use Unet Loader (GGUF).")
    return dm


def apply_lora(model,path,task,strength=1.):
    import comfy.sd
    import comfy.utils
    dm=validate_backbone(model)
    digest,verified_task=validate_adapter_task(path,task)
    sd=comfy.utils.load_torch_file(str(path),safe_load=True)
    down_keys=[k for k in sd if ".lora_A." in k]
    if not down_keys:
        raise ValueError("Expected the released FLUX.2 PEFT UltraTex LoRA")
    config_path=Path(path).with_name("adapter_config.json")
    config=json.loads(config_path.read_text()) if config_path.is_file() else {}
    if config.get("use_dora") or config.get("use_rslora"):
        raise ValueError("DoRA/rank-stabilized adapters are not the released UltraTex adapter format")
    converted={}
    missing=[]
    modules=dict(dm.named_modules())
    paired=set()
    for key in down_keys:
        up_key=key.replace(".lora_A.",".lora_B.")
        if up_key not in sd:
            raise ValueError(f"Missing LoRA B tensor for {key}")
        paired.add(up_key)
        module_name=key.removeprefix("base_model.model.").split(".lora_A.")[0]
        module=modules.get(module_name)
        down,up=sd[key],sd[up_key]
        if module is None or not hasattr(module,"in_features"):
            missing.append(module_name)
            continue
        if down.ndim!=2 or up.ndim!=2 or down.shape[0]!=up.shape[1] or down.shape[1]!=module.in_features or up.shape[0]!=module.out_features:
            raise ValueError(f"LoRA shape mismatch at {module_name}; checkpoint/backbone mismatch")
        target=f"diffusion_model.{module_name}"
        converted[target+".lora_down.weight"]=down
        converted[target+".lora_up.weight"]=up
        alpha=config.get("alpha_pattern",{}).get(module_name,config.get("lora_alpha",down.shape[0]))
        converted[target+".alpha"]=torch.tensor(float(alpha))
    orphan=[k for k in sd if ".lora_B." in k and k not in paired]
    if missing or orphan:
        raise ValueError(f"Adapter has unmatched modules/pairs: {(missing+orphan)[:8]}")
    patched,_=comfy.sd.load_lora_for_models(model,None,converted,float(strength),0.)
    expected={f"diffusion_model.{k.removeprefix('base_model.model.').split('.lora_A.')[0]}.weight" for k in down_keys}
    actual={k for k in expected if len(patched.patches.get(k,()))>len(model.patches.get(k,()))}
    if strength!=0 and actual!=expected:
        raise ValueError(f"Only {len(actual)}/{len(expected)} adapter layers matched. Refusing partial UltraTex conditioning.")
    info=ModelInfo(object_identity(model.model),task,digest,float(strength))
    patched.set_attachments("utx_model_info",info)
    report={"stage":"apply_lora","task":task,"sha256":digest,"source":str(path),
            "matched_layers":len(actual),"expected_layers":len(expected),"strength":strength,
            "alpha_source":"adapter_config.json" if config else "released UltraTex convention: alpha=rank",
            "task_verification":"published SHA-256" if digest in _KNOWN_ADAPTER_TASKS else ("filename hint" if verified_task else "user-selected; unknown artifact"),
            "weight_path":"GGUF" if any("GGUF" in t.__name__.upper() for t in type(model).__mro__) else "native ComfyUI"}
    return patched,info,report


def foreground_decoder(vae,path):
    import comfy.utils
    if getattr(vae,"latent_channels",None)!=128:
        raise ValueError("Foreground decoder requires the matching FLUX.2 VAE (128 packed latent channels)")
    state=comfy.utils.load_torch_file(str(path),safe_load=True)
    if "state_dict" in state and isinstance(state["state_dict"],dict):
        state=state["state_dict"]
    base=vae.get_sd()
    patches={}
    for key,value in state.items():
        target=key if key.startswith(("decoder.","post_quant_conv.")) else "decoder."+key
        if target not in base or tuple(base[target].shape)!=tuple(value.shape):
            raise ValueError(f"Foreground decoder does not match VAE tensor {target}")
        patches[target]=("set",(value.detach().cpu(),))
    expected={key for key in base if key.startswith("decoder.")}
    if not expected.issubset(patches):
        raise ValueError(f"Incomplete foreground decoder: {len(expected-set(patches))} decoder tensors missing")
    # Patcher clones share read-only base storage, while replacement weights are branch-local.
    result=copy.copy(vae)
    result.patcher=vae.patcher.clone()
    accepted=result.patcher.add_patches(patches)
    if set(accepted)!=set(patches):
        raise ValueError("Installed VAE patcher rejected foreground-decoder keys")
    result.utx_encoder_id=getattr(vae,"utx_encoder_id",object_identity(vae.first_stage_model))
    result.utx_decoder_hash=file_hash(path)
    return result,{"stage":"foreground_decoder","sha256":result.utx_decoder_hash,
                   "replaced_tensors":len(patches),"shared_base_storage":True}
