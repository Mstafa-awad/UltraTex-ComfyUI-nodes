"""Read-only Windows per-process DXGI budgets; no driver/global policy changes."""
from __future__ import annotations
import os


def windows_budget(device_name):
    if os.name!='nt':return None
    import ctypes as c
    import uuid
    class GUID(c.Structure):_fields_=[('data',c.c_ubyte*16)]
    class LUID(c.Structure):_fields_=[('low',c.c_uint32),('high',c.c_int32)]
    class Desc(c.Structure):
        _fields_=[('name',c.c_wchar*128),('vendor',c.c_uint32),('device',c.c_uint32),('subsys',c.c_uint32),('revision',c.c_uint32),('dedicated',c.c_size_t),('system',c.c_size_t),('shared',c.c_size_t),('luid',LUID),('flags',c.c_uint32)]
    class Info(c.Structure):_fields_=[('budget',c.c_uint64),('usage',c.c_uint64),('available_reservation',c.c_uint64),('reservation',c.c_uint64)]
    def guid(s):return GUID((c.c_ubyte*16).from_buffer_copy(uuid.UUID(s).bytes_le))
    def call(ptr,index,restype,*argtypes):
        address=c.cast(ptr,c.POINTER(c.POINTER(c.c_void_p))).contents[index]
        return c.WINFUNCTYPE(restype,c.c_void_p,*argtypes)(address)
    factory=c.c_void_p();adapters=[]
    try:
        dll=c.WinDLL('dxgi.dll')
        create=dll.CreateDXGIFactory1;create.argtypes=[c.POINTER(GUID),c.POINTER(c.c_void_p)];create.restype=c.c_int32
        iid=guid('770aae78-f26f-4dba-a829-253c83d1b387')
        if create(c.byref(iid),c.byref(factory))<0:return None
        index=0
        while True:
            adapter=c.c_void_p()
            if call(factory,12,c.c_int32,c.c_uint32,c.POINTER(c.c_void_p))(factory,index,c.byref(adapter))<0:break
            adapters.append(adapter);index+=1;desc=Desc()
            if call(adapter,10,c.c_int32,c.POINTER(Desc))(adapter,c.byref(desc))<0:continue
            if desc.vendor!=0x10de or desc.name.casefold()!=device_name.casefold():continue
        matches=[]
        for adapter in adapters:
            desc=Desc()
            if call(adapter,10,c.c_int32,c.POINTER(Desc))(adapter,c.byref(desc))>=0 and desc.name.casefold()==device_name.casefold():matches.append((adapter,desc))
        if len(matches)!=1:return None
        adapter,desc=matches[0];a3=c.c_void_p();iid3=guid('645967a4-1392-4310-a798-8053ce3e93fd')
        if call(adapter,0,c.c_int32,c.POINTER(GUID),c.POINTER(c.c_void_p))(adapter,c.byref(iid3),c.byref(a3))<0:return None
        try:
            local,shared=Info(),Info()
            query=call(a3,14,c.c_int32,c.c_uint32,c.c_uint32,c.POINTER(Info))
            if query(a3,0,0,c.byref(local))<0 or query(a3,0,1,c.byref(shared))<0:return None
            return dict(local_budget=local.budget,local_usage=local.usage,shared_usage=shared.usage,
                        dedicated=desc.dedicated,luid=f'{desc.luid.high&0xffffffff:08x}:{desc.luid.low:08x}')
        finally:call(a3,2,c.c_uint32)(a3)
    except (OSError,ValueError,AttributeError):return None
    finally:
        for adapter in adapters:call(adapter,2,c.c_uint32)(adapter)
        if factory:call(factory,2,c.c_uint32)(factory)
