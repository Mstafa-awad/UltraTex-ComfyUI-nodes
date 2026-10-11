"""Compare chunked and native ComfyUI blocks; no model weights are needed."""
import argparse
import json
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--comfy-root',required=True)
p.add_argument('--report',required=True)
args=p.parse_args()
sys.path[:0]=[args.comfy_root,str(Path(__file__).resolve().parents[2])]
sys.argv=[sys.argv[0]]

import torch
from comfy.ldm.flux.layers import SingleStreamBlock
from comfy.ldm.flux.math import rope
from comfyui_ultratex_modular.vendor.optim import chunked_single_block
from comfyui_ultratex_modular.extension import ApplyTextureLora
from comfyui_ultratex_modular.core.models import select_adapter_name
import folder_paths

torch.set_num_threads(4)
torch.manual_seed(902)
results=[]
with torch.inference_mode():
    for yak,silu in ((False,False),(False,True),(True,False)):
        block=SingleStreamBlock(64,4,mlp_ratio=2.,yak_mlp=yak,mlp_silu_act=silu,
                                dtype=torch.float32,device='cpu',operations=torch.nn)
        x=torch.randn(1,53,64)
        vec=torch.randn(1,64)
        pe=rope(torch.arange(53)[None],16,10000)[:,None]
        expected=block(x.clone(),vec,pe)
        actual=chunked_single_block(block,17)({'img':x.clone(),'vec':vec,'pe':pe}, {})['img']
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        results.append({'yak':yak,'silu':silu,'max_abs_difference':float((actual-expected).abs().max())})
names={task:select_adapter_name(folder_paths.get_filename_list('loras'),task,
       lambda name:folder_paths.get_full_path_or_raise('loras',name)) for task in ('albedo','mr')}
assert ApplyTextureLora.validate_inputs(names['albedo'],'albedo') is True
rejected=ApplyTextureLora.validate_inputs(names['mr'],'albedo')
assert isinstance(rejected,str) and 'Adapter/task mismatch' in rejected,rejected
result={'single_block_native_parity':results,'v3_wrong_adapter_validation':rejected}
# Exercise ComfyUI's actual prompt validator, not only the node's direct method.
import asyncio
import nodes
import execution
class ModelFixture:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{}}
    RETURN_TYPES=('MODEL',)
    FUNCTION='run'
    def run(self):raise AssertionError('Validation must not execute/load a model')
class OutputFixture:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'model':('MODEL',)}}
    RETURN_TYPES=()
    OUTPUT_NODE=True
    FUNCTION='run'
    def run(self,model):raise AssertionError('Validation must not execute the graph')
nodes.NODE_CLASS_MAPPINGS.update(UTXValidationModelFixture=ModelFixture,UTXValidationOutputFixture=OutputFixture,
                               UTXApplyTextureLora=ApplyTextureLora)
prompt={'1':{'class_type':'UTXValidationModelFixture','inputs':{}},
        '2':{'class_type':'UTXApplyTextureLora','inputs':{'model':['1',0],
             'lora_name':names['albedo'],'task':'albedo','strength':1.}},
        '3':{'class_type':'UTXValidationOutputFixture','inputs':{'model':['2',0]}}}
accepted=asyncio.run(execution.validate_prompt('correct-adapter',prompt,None))
assert accepted[0],accepted
prompt['2']['inputs']['lora_name']=names['mr']
blocked=asyncio.run(execution.validate_prompt('incorrect-adapter',prompt,None))
assert not blocked[0] and 'Adapter/task mismatch' in str(blocked),blocked
result['comfy_prompt_validation']={'correct_accepted':True,'mismatch_rejected_before_execution':True}
Path(args.report).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
