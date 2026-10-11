"""Generate importable UI workflows using the installed ComfyUI node schemas.

This only writes into this package's examples directory. It does not install nodes.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--comfy-root",required=True)
args=p.parse_args()
root=Path(args.comfy_root)
package=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(package.parent)]
sys.argv=[sys.argv[0]]

import nodes
import folder_paths
from comfy_extras.nodes_load_3d import Load3DAdvanced
from comfy_extras.nodes_save_3d import SaveGLB
from comfyui_ultratex_modular.extension import NODES
from comfyui_ultratex_modular.core.models import select_adapter_name

adapter_names={task:select_adapter_name(folder_paths.get_filename_list('loras'),task,
               lambda name:folder_paths.get_full_path_or_raise('loras',name)) for task in ('albedo','mr')}

gguf=root/"custom_nodes/ComfyUI-GGUF"
spec=importlib.util.spec_from_file_location("utx_workflow_gguf",gguf/"__init__.py",submodule_search_locations=[str(gguf)])
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
registry=dict(nodes.NODE_CLASS_MAPPINGS,**module.NODE_CLASS_MAPPINGS)
registry.update({cls.define_schema().node_id:cls for cls in [*NODES,Load3DAdvanced,SaveGLB]})


def info(name):
    cls=registry[name]
    if hasattr(cls,"define_schema"):
        value=cls.define_schema().get_v1_info(cls)
        return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else vars(value)
    return {"input":cls.INPUT_TYPES(),"output":cls.RETURN_TYPES,
            "output_name":getattr(cls,"RETURN_NAMES",cls.RETURN_TYPES),"display_name":name}


class Graph:
    def __init__(self):
        self.nodes=[]
        self.links=[]
        self.api={}
        self.definitions={}
        self.repaired={}

    def add(self,kind,x,y,title=None,**values):
        definition=info(kind)
        node_id=len(self.nodes)+1
        inputs=[]
        widgets=[]
        api_inputs={}
        for group in ("required","optional"):
            for name,entry in definition["input"].get(group,{}).items():
                typ=entry[0]
                options=entry[1] if len(entry)>1 else {}
                is_widget=isinstance(typ,list) or typ in ("INT","FLOAT","STRING","BOOLEAN","COMBO","LOAD_3D")
                if is_widget and not options.get("forceInput",False):
                    choices=typ if isinstance(typ,list) else options.get("options",[])
                    default=choices[0] if choices else {"INT":0,"FLOAT":0.,"STRING":"","BOOLEAN":False,"LOAD_3D":{}}.get(typ,"")
                    value=values.pop(name,options.get("default",default))
                    widgets.append(value)
                    api_inputs[name]=value
                    if options.get("control_after_generate"):
                        widgets.append("fixed")
                else:
                    inputs.append({"name":name,"type":typ,"link":None,**({"shape":7} if group=="optional" else {})})
        if values:
            raise ValueError(f"Unknown widgets on {kind}: {values}")
        outputs=[{"name":label,"type":typ,"links":[],"slot_index":i} for i,(typ,label) in
                 enumerate(zip(definition["output"],definition.get("output_name",definition["output"])))]
        node={"id":node_id,"type":kind,"pos":[x,y],"size":[330,max(100,75+len(inputs)*22+len(widgets)*30)],
              "flags":{},"order":len(self.nodes),"mode":0,"inputs":inputs,"outputs":outputs,
              "properties":{"Node name for S&R":kind},"widgets_values":widgets}
        if title:
            node["title"]=title
        self.nodes.append(node)
        self.api[str(node_id)]={"class_type":kind,"inputs":api_inputs}
        self.definitions[node_id]=definition
        return node_id

    def link(self,source,slot,target,name):
        src=self.nodes[source-1]
        dst=self.nodes[target-1]
        index=next(i for i,item in enumerate(dst["inputs"]) if item["name"]==name)
        typ=src["outputs"][slot]["type"]
        allowed=dst["inputs"][index]["type"].split(",")
        assert typ in allowed or "*" in allowed,(src["type"],typ,dst["type"],allowed)
        number=len(self.links)+1
        assert dst["inputs"][index]["link"] is None
        src["outputs"][slot]["links"].append(number)
        dst["inputs"][index]["link"]=number
        self.links.append([number,source,slot,target,index,typ])
        self.api[str(target)]["inputs"][name]=[str(source),slot]

    def write(self,name,note):
        # Verify every required socket and widget in the actual installed schemas.
        for ident,node in self.api.items():
            for key in self.definitions[int(ident)]["input"].get("required",{}):
                assert key in node["inputs"],(node["class_type"],key)
        note_node={"id":len(self.nodes)+1,"type":"Note","pos":[0,-360],"size":[900,280],"flags":{},
                   "order":0,"mode":0,"properties":{},"widgets_values":[note],"color":"#432","bgcolor":"#653"}
        document={"last_node_id":note_node["id"],"last_link_id":len(self.links),"nodes":[*self.nodes,note_node],
                  "links":self.links,"groups":[],"config":{},"extra":{"ds":{"scale":.65,"offset":[40,260]},
                  "ultratex_modular":{"version":"0.1.2","description":note}},"version":.4}
        folder=package/"examples"
        folder.mkdir(exist_ok=True)
        (folder/(name+".json")).write_text(json.dumps(document,indent=2),encoding="utf-8")
        # An API template is useful for scripted runs; input filenames are placeholders.
        api_folder=folder/"api"
        api_folder.mkdir(exist_ok=True)
        (api_folder/(name+".api.json")).write_text(json.dumps(self.api,indent=2),encoding="utf-8")
        return len(self.nodes)


def mesh_input(g):
    loader=g.add("Load3DAdvanced",0,0,model_file="none",width=512,height=512)
    mesh=g.add("UTXFileToMesh",400,0)
    g.link(loader,0,mesh,"file")
    return mesh


def runtime_and_geometry(g,mesh):
    runtime=g.add("UTXConfigureRuntime",0,650,token_chunk=4096,gpu_headroom_mb=768)
    context=g.add("UTXPrepareGeometry",800,0)
    g.link(mesh,0,context,"mesh")
    g.link(runtime,0,context,"runtime")
    return runtime,context


def finish(g,mesh,maps,x=3600,y=0):
    material=g.add("UTXApplyMaterials",x,y)
    g.link(mesh,0,material,"mesh")
    for semantic,(node,slot,meta) in maps.items():
        g.link(g.repaired.get(node,node),slot,material,semantic)
        if meta is not None:
            g.link(meta,1,material,semantic+"_info")
    file=g.add("UTXCreate3DFile",x+400,y)
    g.link(material,0,file,"mesh")
    save=g.add("SaveGLB",x+800,y,filename_prefix="ultratex/textured")
    g.link(file,0,save,"mesh")


def bake(g,atlas,views,runtime,semantic="base_color",job=0,x=3200,y=0):
    node=g.add("UTXBakeMap",x,y,semantic=semantic,job_index=job)
    g.link(atlas,0,node,"atlas")
    g.link(views,0,node,"views")
    g.link(runtime,0,node,"runtime")
    repair=g.add("UTXRepairAtlas",x+350,y+400,mode="nearest_surface",padding=8,max_fill_distance=32)
    g.link(node,0,repair,"image")
    g.link(node,4,repair,"holes")
    g.link(atlas,0,repair,"atlas")
    g.repaired[node]=repair
    save=g.add("SaveImage",x+700,y+400,filename_prefix="ultratex/"+semantic)
    g.link(repair,0,save,"images")
    return node


def generation(name,size=512,count=1,mr=False,native=False,preview=False,weight_format="gguf"):
    g=Graph()
    mesh=mesh_input(g)
    runtime,context=runtime_and_geometry(g,mesh)
    atlas_size=4096 if size==2048 else (2048 if size==1024 else 1024)
    plan=g.add("UTXPlanViews",1200,0,render_size=2048 if size==1024 else size,sample_size=size)
    g.link(context,0,plan,"context")
    gb=g.add("UTXRenderGBuffers",1600,0)
    for n,s,d in ((context,0,"context"),(plan,0,"plan"),(runtime,0,"runtime")):
        g.link(n,s,gb,d)
    image=g.add("LoadImage",400,600,image="reference.png")
    reference=g.add("UTXPrepareReference",800,450)
    for n,s,d in ((image,0,"image"),(image,1,"mask"),(gb,0,"gbuffer"),(runtime,0,"runtime")):
        g.link(n,s,reference,d)
    vae=g.add("VAELoader",800,850,vae_name="flux2-vae.safetensors")
    fgvae=g.add("UTXApplyForegroundDecoder",1200,850,decoder_name="ultratex_flux2_decoder.pt")
    g.link(vae,0,fgvae,"vae")
    conditions=g.add("UTXEncodeConditions",1200,400)
    for n,s,d in ((vae,0,"vae"),(reference,0,"reference"),(gb,0,"gbuffer"),(runtime,0,"runtime")):
        g.link(n,s,conditions,d)
    tokens=g.add("UTXBuildForegroundTokens",1600,400)
    g.link(conditions,0,tokens,"conditions")
    g.link(runtime,0,tokens,"runtime")
    clip=g.add("CLIPLoaderGGUF",1200,1120,clip_name="Qwen3-8B-Q6_K.gguf",type="flux2")
    text=g.add("UTXTextEncode",1600,1050,text="")
    g.link(clip,0,text,"clip")
    work=g.add("UTXBuildWorkset",2000,400,count=count)
    for n,s,d in ((tokens,0,"tokens"),(text,0,"positive"),(runtime,0,"runtime")):
        g.link(n,s,work,d)
    if weight_format=="gguf":
        model=g.add("UnetLoaderGGUF",1600,700,unet_name="flux-2-klein-base-9b-Q4_0.gguf")
    else:
        model=g.add("UNETLoader",1600,700,unet_name="flux-2-klein-base-9b-fp8.safetensors",weight_dtype="default")
    albedo=g.add("UTXApplyTextureLora",2000,800,lora_name=adapter_names['albedo'])
    g.link(model,0,albedo,"model")
    mr_model=None
    if mr:
        mr_model=g.add("UTXApplyTextureLora",2000,1150,lora_name=adapter_names['mr'],task="mr")
        g.link(model,0,mr_model,"model")
        mr_work=g.add("UTXBuildWorkset",2400,800,task="mr")
        for n,s,d in ((tokens,0,"tokens"),(text,0,"positive"),(runtime,0,"runtime"),(work,0,"previous")):
            g.link(n,s,mr_work,d)
        work=mr_work
    sample=g.add("UTXSampleLatents",2400,0,steps=5 if preview else 25,guidance=1.)
    for n,s,d in ((albedo,0,"model"),(work,0,"workset"),(runtime,0,"runtime")):
        g.link(n,s,sample,d)
    if mr_model:
        g.link(mr_model,0,sample,"material_model")
    decode=g.add("UTXDecodeViews",2800,0)
    for n,s,d in ((fgvae,0,"vae"),(sample,0,"samples"),(runtime,0,"runtime")):
        g.link(n,s,decode,d)
    saved=g.add("UTXSaveViews",2800,450)
    g.link(decode,0,saved,"views")
    inspect=g.add("UTXInspectViews",2800,700,preview_size=512)
    g.link(decode,0,inspect,"views")
    preview=g.add("PreviewImage",3200,700)
    g.link(inspect,3,preview,"images")
    atlas=g.add("UTXPrepareAtlasProjection",2800,1100,atlas_size=atlas_size)
    for n,s,d in ((context,0,"context"),(gb,0,"gbuffer"),(runtime,0,"runtime")):
        g.link(n,s,atlas,d)
    base=bake(g,atlas,decode,runtime,x=3200,y=0)
    maps={"base_color":(base,0,base)}
    if mr:
        for semantic,y in (("roughness",1100),("metallic",1900)):
            channel=bake(g,atlas,decode,runtime,semantic,1,x=3200,y=y)
            maps[semantic]=(channel,0,channel)
    finish(g,mesh,maps)
    note=f"UltraTex Modular {size}px six-view generation. Select a single-material UV-mapped GLB and a transparent front-view reference. Check model filenames. Each job jointly samples SIX views, saved to a portable bundle for rebaking. The atlas is {atlas_size}px. Bake uses smooth view preference and depth-edge rejection. Surface fill and 8px gutters preserve observed texels. Hidden details require additional views. All sampling finishes before decoding."
    if preview:
        note+=" Five steps is only a pipeline preview. The undistilled base model needs more steps for quality; start with 25."
    note+=f" Local verified albedo selection: {adapter_names['albedo']}; albedo SHA-256 starts 3b191517. Trust the hash if your filenames are swapped."
    if weight_format!="gguf":
        note+=" OPTIONAL native FP8 loader: requires your separately installed BASE 9B FP8 weights. This path has not been benchmarked locally; no FP8 weights were available. The Qwen GGUF encoder can be retained independently."
    if count>1:
        note+=" This example requests 20 variations (120 views). It saves the complete view bundle; preview and bake default to job 0. Adjust job_index to inspect/bake another job. Watch disk and RAM limits."
    if mr:
        note+=" Job 0 is albedo, job 1 is MR. Both model branches MUST come from the same backbone loader. Roughness=G, metallic=B."
    if native:
        note+=" NATIVE MESH: connect your geometry node's native MESH output to Prepare Geometry.mesh and Apply Materials.mesh, replacing both File to Mesh connections. The GLB input is a runnable example adapter. Independent IMAGE maps can be connected to Apply Materials."
    if size==2048:
        note+=" 25 steps is a quality starting point, NOT a validated optimum. Native 2K is slow on the tested 8GB GPU; first verify the 512px workflow."
    return g.write(name,note)


def bake_only():
    g=Graph()
    mesh=mesh_input(g)
    runtime,context=runtime_and_geometry(g,mesh)
    views=g.add("UTXLoadViews",1200,400)
    atlas=g.add("UTXPrepareAtlasProjection",1600,0,atlas_size=4096)
    for n,s,d in ((context,0,"context"),(views,1,"gbuffer"),(runtime,0,"runtime")):
        g.link(n,s,atlas,d)
    base=bake(g,atlas,views,runtime,x=2000,y=0)
    finish(g,mesh,{"base_color":(base,0,base)},x=2400)
    return g.write("04_bake_saved_views", "No generation models are loaded. Select the original single-material UV mesh and set manifest_path to a saved views.utx.json. You may change the UV layout, atlas size and bake settings while preserving the original positions, faces and normals. For a different job, set the same job_index in Load View Bundle and Bake Map. A bundle contains its calibrated cameras and depths.")


def materials_only():
    g=Graph()
    mesh=mesh_input(g)
    maps={}
    for i,name in enumerate(("base_color","roughness","metallic")):
        node=g.add("LoadImage",800,400*i,image=name+".png")
        maps[name]=(node,0,None)
    finish(g,mesh,maps,x=1200)
    return g.write("05_materials_only", "No diffusion or VAE required. Select a UV-mapped GLB and maps authored for exactly those UVs. Disconnect a map to inherit the existing channel. IMAGE maps from UltraTex, Trellis2, Pixal3D or other nodes can connect directly. Optional AO, tangent normal, emissive and opacity sockets are independent. Use OpenGL normals, or select DirectX to flip green. Opacity requires MASK or BLEND and UTX Create 3D File before Save 3D.")


def external_views():
    g=Graph()
    mesh=mesh_input(g)
    runtime,context=runtime_and_geometry(g,mesh)
    plan=g.add("UTXPlanViews",1200,0,render_size=2048,sample_size=1024)
    g.link(context,0,plan,"context")
    gb=g.add("UTXRenderGBuffers",1600,0)
    for n,s,d in ((context,0,"context"),(plan,0,"plan"),(runtime,0,"runtime")):
        g.link(n,s,gb,d)
    previous=None
    for i,name in enumerate(("front","left","back","right","top","bottom")):
        image=g.add("LoadImage",800,400+i*400,image=name+".png")
        if previous is None:
            previous=image
        else:
            batch=g.add("ImageBatch",1200,400+i*400)
            g.link(previous,0,batch,"image1")
            g.link(image,0,batch,"image2")
            previous=batch
    views=g.add("UTXViewsFromImages",1600,500)
    for n,s,d in ((previous,0,"images"),(gb,0,"gbuffer"),(runtime,0,"runtime")):
        g.link(n,s,views,d)
    atlas=g.add("UTXPrepareAtlasProjection",2000,0,atlas_size=4096)
    for n,s,d in ((context,0,"context"),(gb,0,"gbuffer"),(runtime,0,"runtime")):
        g.link(n,s,atlas,d)
    base=bake(g,atlas,views,runtime,x=2400,y=0)
    finish(g,mesh,{"base_color":(base,0,base)},x=2800)
    return g.write("08_external_views_bake_only", "No diffusion or VAE required. Supply six 1024x1024 images in front,left,back,right,top,bottom order, rendered/edited from THIS canonical view rig and unchanged mesh. Arbitrary photographs are not calibrated by this node. Match sample_size to your image dimensions. Depth visibility comes from the linked G-buffers. This example bakes base color; scalar MR inputs require task=mr and the matching Bake Map semantic.")


def simple_generation(view_count=6):
    g=Graph()
    mesh=mesh_input(g)
    image=g.add('LoadImage',0,550,image='reference.png')
    setup=g.add('UTXPrepareTexture',800,0,view_count=view_count,mask_polarity='background')
    g.link(mesh,0,setup,'mesh');g.link(image,0,setup,'reference');g.link(image,1,setup,'mask')
    model=g.add('UnetLoaderGGUF',400,400,unet_name='flux-2-klein-base-9b-Q4_0.gguf')
    vae=g.add('VAELoader',400,600,vae_name='flux2-vae.safetensors')
    clip=g.add('CLIPLoaderGGUF',400,800,clip_name='Qwen3-8B-Q6_K.gguf',type='flux2')
    text=g.add('UTXTextEncode',800,700,text='');g.link(clip,0,text,'clip')
    generate=g.add('UTXGenerateTextureViews',1200,0,lora_name=adapter_names['albedo'],decoder_name='ultratex_flux2_decoder.pt')
    for node,name in ((setup,'setup'),(model,'model'),(vae,'vae'),(text,'positive')):g.link(node,0,generate,name)
    preview=g.add('PreviewImage',1200,500,title='All generated views');g.link(generate,2,preview,'images')
    saveviews=g.add('UTXSaveViews',1200,1000);g.link(generate,0,saveviews,'views')
    bake=g.add('UTXBakeTexture',1600,0);g.link(setup,0,bake,'setup');g.link(generate,0,bake,'views')
    save=g.add('SaveGLB',2000,0,filename_prefix='ultratex/textured');g.link(bake,2,save,'mesh')
    imageout=g.add('SaveImage',2000,500,filename_prefix='ultratex/base_color');g.link(bake,1,imageout,'images')
    return g.write('00_simple_'+str(view_count)+'_views',
        f'Start here: Prepare Texture -> Generate Views -> Bake Texture. Connect your UV-mapped MESH at Prepare Texture.mesh (or select a GLB with existing UVs). Your UV code stays external. {view_count} views, 1024px generation, 2048px texture, 25 steps. Change view_count to 6 for one pass or 12 for two passes (45-degree second rig, roughly twice the generation work). Every view appears in the labeled contact sheet; Generate Views also outputs an IMAGE batch. Save View Bundle retains full-resolution data. Hidden surface fill approximates colors, not unseen detail. No Diagnostics node is needed.')

counts={}
counts['00_simple_6_views']=simple_generation(6)
counts['00_simple_12_views']=simple_generation(12)
counts["01_preview_512"]=generation("01_preview_512",preview=True)
counts["02_native_2k_4k_atlas"]=generation("02_native_2k_4k_atlas",2048)
counts["03_albedo_and_mr"]=generation("03_albedo_and_mr",512,mr=True)
counts["04_bake_saved_views"]=bake_only()
counts["05_materials_only"]=materials_only()
counts["06_twenty_variations"]=generation("06_twenty_variations",512,count=20,preview=True)
counts["07_native_mesh_integration"]=generation("07_native_mesh_integration",512,native=True)
counts["08_external_views_bake_only"]=external_views()
counts["09_1k_views_2k_atlas"]=generation("09_1k_views_2k_atlas",1024)
counts["10_native_fp8_1k_views_2k_atlas"]=generation("10_native_fp8_1k_views_2k_atlas",1024,weight_format="native")
(package/"examples"/"validation.json").write_text(json.dumps({"schema_validated":True,"node_counts":counts,
    "notes":"All required sockets/widgets and link types checked against installed schemas. Input filenames require selection. UI browser import and artistic outputs have not been tested."},indent=2),encoding="utf-8")
print(json.dumps(counts,indent=2))
