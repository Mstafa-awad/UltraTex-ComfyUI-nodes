"""Portable view bundles. JSON + NPY only; never pickle or executable serialization."""
from __future__ import annotations

import dataclasses
import json
import shutil
import uuid
from pathlib import Path

from . import contracts as c
from .runtime import atomic_json, interrupt

_TYPES={x.__name__:x for x in (c.ArrayRef,c.ViewPlan,c.GBuffer,c.ViewSet,c.ViewCollection,c.ImageTransform)}


def save_views(views,directory,prefix="ultratex_views"):
    root=Path(directory).resolve()
    parent=(root/(prefix.strip() or "ultratex_views")).resolve()
    if parent==root or not parent.is_relative_to(root):
        raise ValueError("View bundle prefix must stay inside the output directory")
    folder=parent.with_name(parent.name+"_"+uuid.uuid4().hex[:12])
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"arrays").mkdir()
    copied=set()
    def encode(value):
        interrupt()
        if isinstance(value,c.ArrayRef):
            name=value.digest+".npy"
            if name not in copied:
                shutil.copyfile(value.path,folder/"arrays"/name)
                copied.add(name)
            return {"type":"ArrayRef","path":"arrays/"+name,"shape":list(value.shape),"dtype":value.dtype,"digest":value.digest}
        if dataclasses.is_dataclass(value) and type(value).__name__ in _TYPES:
            return {"type":type(value).__name__,**{f.name:encode(getattr(value,f.name)) for f in dataclasses.fields(value)
                if not (isinstance(value,c.ViewSet) and f.name in ("sampling","original_images","original_masks") and not getattr(value,f.name))}}
        if isinstance(value,(tuple,list)):
            return [encode(x) for x in value]
        return value
    # A failed/cancelled write leaves no completed manifest and cannot be mistaken for a valid bundle.
    document=encode(views)
    path=folder/"views.utx.json"
    atomic_json(path,{"version":2 if any(v.sampling for v in views.sets) else 1,"complete":True,"views":document})
    return str(path)


def load_views(manifest):
    path=Path(manifest).expanduser().resolve()
    document=json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") not in (1,2) or document.get("complete") is not True:
        raise ValueError("Unsupported or incomplete UltraTex view bundle")
    def decode(value):
        if isinstance(value,list):
            return tuple(decode(x) for x in value)
        if isinstance(value,dict):
            name=value.get("type")
            if name not in _TYPES:
                raise ValueError(f"Unsupported bundle record {name}")
            fields={k:decode(v) for k,v in value.items() if k!="type"}
            if name=="ArrayRef":
                array_path=(path.parent/fields["path"]).resolve()
                if not array_path.is_relative_to(path.parent):
                    raise ValueError("Bundle array path escapes its directory")
                fields["path"]=str(array_path)
            result=_TYPES[name](**fields)
            if isinstance(result,c.ArrayRef):
                result.read()
            return result
        return value
    views=decode(document["views"])
    if not isinstance(views,c.ViewCollection):
        raise ValueError("Expected a view collection")
    for view in views.sets:
        if len(view.images)!=6 or len(view.masks)!=6 or len(view.plan.matrices)!=6:
            raise ValueError("View bundle must contain complete six-view sets")
        if view.sampling and len(view.sampling) != 6:
            raise ValueError("Enhanced bundle needs six crop transforms per rig")
        if view.sampling and (len(view.original_images)!=6 or len(view.original_masks)!=6):
            raise ValueError("Enhanced bundle is missing its original images/masks for rollback")
        if view.sampling:
            for image,original,tr in zip(view.images,view.original_images,view.sampling):
                if image.shape[:2]!=(tr.output_size,tr.output_size) or original.shape[:2]!=(tr.original_height,tr.original_width):
                    raise ValueError("Enhanced bundle image dimensions disagree with its transform")
    return views
