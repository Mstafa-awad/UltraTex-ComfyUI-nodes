"""Exercise the real local Qwen3 GGUF loader and UTX text node, without a DiT."""
import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--comfy-root",required=True)
p.add_argument("--report",required=True)
args=p.parse_args()
root=Path(args.comfy_root)
sys.path[:0]=[str(root),str(Path(__file__).resolve().parents[2])]
sys.argv=[sys.argv[0]]
import torch
import comfy.model_management as mm
from comfyui_ultratex_modular.extension import TextEncode

directory=root/"custom_nodes/ComfyUI-GGUF"
spec=importlib.util.spec_from_file_location("utx_text_gguf",directory/"__init__.py",submodule_search_locations=[str(directory)])
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
start=time.perf_counter()
clip=module.NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]().load_clip("Qwen3-8B-Q6_K.gguf","flux2")[0]
conditioning=TextEncode.execute(clip,"")[0]
embedding,metadata=conditioning[0]
assert tuple(embedding.shape)==(1,512,12288),tuple(embedding.shape)
assert torch.isfinite(embedding).all()
report={"kind":"actual empty-prompt Qwen3-8B Q6_K encoding via UTXTextEncode",
        "shape":list(embedding.shape),"metadata":sorted(metadata),"finite":True,"wall_seconds":time.perf_counter()-start}
Path(args.report).write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
mm.unload_all_models()
del clip
gc.collect()
