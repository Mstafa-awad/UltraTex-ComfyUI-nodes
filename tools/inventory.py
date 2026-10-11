"""Record local artifact hashes and runtime versions without loading model weights."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--comfy-root",required=True)
p.add_argument("--output",required=True)
args=p.parse_args()
root=Path(args.comfy_root)
files=[
    ("diffusion_models/flux-2-klein-base-9b-Q4_0.gguf","FLUX.2 Klein base 9B GGUF Q4_0",
     "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B","BFL FLUX non-commercial model license; quantizer provenance not established from local filename"),
    ("loras/ultratex_flux2.safetensors","UltraTex FLUX.2 albedo PEFT adapter",
     "https://www.modelscope.ai/models/Yibo-Zhang/UltraTex","UltraTex code is MIT; independent artifact-specific grant not established, base-model obligations remain"),
    ("loras/ultratex_flux2_mr.safetensors","UltraTex FLUX.2 metallic-roughness PEFT adapter",
     "https://www.modelscope.ai/models/Yibo-Zhang/UltraTex","UltraTex code is MIT; independent artifact-specific grant not established, base-model obligations remain"),
    ("ultratex/ultratex_flux2_decoder.pt","UltraTex FLUX.2 foreground decoder",
     "https://www.modelscope.ai/models/Yibo-Zhang/UltraTex","Independent artifact-specific grant not established from local file"),
    ("vae/flux2-vae.safetensors","FLUX.2 VAE",
     "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B","Check the license attached to the exact VAE distribution; local source provenance unavailable"),
    ("text_encoders/Qwen3-8B-Q6_K.gguf","Qwen3-8B GGUF text encoder for flux2",
     "https://huggingface.co/Qwen/Qwen3-8B","Qwen3-8B upstream Apache-2.0; quantized artifact provider not established from local filename"),
]
models=[]
for relative,label,source,terms in files:
    path=root/"models"/relative
    entry={"path_under_models":relative,"component":label,"upstream_reference":source,"terms":terms,
           "redistributed":False,"source_verification":"reference family only; no claim of authenticated local download provenance"}
    if path.is_file():
        h=hashlib.sha256()
        with path.open("rb") as stream:
            while data:=stream.read(8*2**20):
                h.update(data)
        entry.update(size_bytes=path.stat().st_size,sha256=h.hexdigest())
    else:
        entry["status"]="not found"
    models.append(entry)
versions={}
for name in ("torch","numpy","Pillow","scipy","einops","moderngl","glcontext","psutil","triton-windows","gguf"):
    try:
        versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name]=None
def revision(path):
    try:
        return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return None
result={"package":"UltraTex Modular","version":"0.1.0","measured_on":"2026-10-10",
        "python":platform.python_version(),"dependencies":versions,
        "comfyui_revision":revision(root),"gguf_revision":revision(root/"custom_nodes/ComfyUI-GGUF"),
        "vendored_wrapper_revision":"8137da52ed8352305cb7825b39c36a250bf2507a",
        "models":models,"license_reference":"https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md",
        "model_use_and_outputs":"Model-use and output-use permissions are separate; consult the referenced current license. The node pack adds no output ownership or watermark terms.",
        "runtime_note":"Installed Triton 3.8.0.post28 passed this machine's sparse tests with torch 2.8.0+cu128. This differs from the upstream supported minor-version matrix; it is not a universal compatibility claim."}
Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({"output":args.output,"models_hashed":sum("sha256" in m for m in models)},indent=2))
