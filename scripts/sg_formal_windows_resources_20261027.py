#!/usr/bin/env python3
"""Read-only Windows RAM/disk producer; metrics are never optimizer progress."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, shutil, signal, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/2026-10-27-generator-formal'
DEST=RUN/'runtime/windows-resources.json'
STOP=False
class MemoryStatus(ctypes.Structure):
 _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong),('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),('ullAvailExtendedVirtual',ctypes.c_ulonglong)]
def sample():
 if os.name!='nt':raise PermissionError('Windows native read-only probe only')
 x=MemoryStatus();x.dwLength=ctypes.sizeof(x)
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(MemoryStatus)];kernel.GlobalMemoryStatusEx.restype=ctypes.c_int
 if not kernel.GlobalMemoryStatusEx(ctypes.byref(x)):raise ctypes.WinError(ctypes.get_last_error())
 disk=shutil.disk_usage('C:/')
 return dict(schema='sg-formal-Windows-physical-metrics-v1',epoch=time.time(),windows_free_gib=x.ullAvailPhys/2**30,windows_total_gib=x.ullTotalPhys/2**30,c_free_gib=disk.free/2**30,c_total_gib=disk.total/2**30,pid=os.getpid(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),probe='GlobalMemoryStatusEx + disk_usage(C:/)',optimizer_progress=False)
def publish(value):
 DEST.parent.mkdir(parents=True,exist_ok=True);tmp=DEST.with_name(DEST.name+f'.tmp-{os.getpid()}-{time.time_ns()}')
 with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,DEST)
def stopped(*_):
 global STOP
 STOP=True

def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['once','watch']);x=a.parse_args()
 signal.signal(signal.SIGINT,stopped);signal.signal(signal.SIGTERM,stopped)
 # Named Windows mutex prevents two local writers without touching system settings.
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p];kernel.CreateMutexW.restype=ctypes.c_void_p
 handle=kernel.CreateMutexW(None,True,'Local\\CodexSGFormal20261027WindowsMetrics')
 if not handle or ctypes.get_last_error()==183:raise PermissionError('Windows metrics producer already owned')
 try:
  while not STOP:
   z=sample();publish(z)
   if x.action=='once':print(json.dumps(z,sort_keys=True));return
   for _ in range(20):
    if STOP or (RUN/'runtime/windows-producer-stop.json').exists():return
    time.sleep(1)
 finally:kernel.ReleaseMutex(ctypes.c_void_p(handle));kernel.CloseHandle(ctypes.c_void_p(handle))
if __name__=='__main__':main()
