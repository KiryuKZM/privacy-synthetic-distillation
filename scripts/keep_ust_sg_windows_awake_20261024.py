#!/usr/bin/env python3
"""Temporary Windows idle-sleep inhibition owned by this live project only."""
import argparse,ctypes,json,os,signal,sys,time
from pathlib import Path
ES_CONTINUOUS=0x80000000
ES_SYSTEM_REQUIRED=0x00000001
TERMINAL={'COMPLETE','COMPLETED','PARTIAL_STOPPED','USER_STOPPED','STOPPED','FAILED','BLOCKED'}
STOP=False
def stop(*_):
    global STOP
    STOP=True
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)
def condition(heartbeat,goal,stops,now,stale=600):
    if STOP:return 'USER_SIGNAL'
    if any(Path(p).exists() for p in stops):return 'PROJECT_STOP_FILE'
    if not Path(goal).is_file():return 'GOAL_STATE_MISSING'
    state=str(load(goal).get('status','')).upper()
    if state in TERMINAL:return 'GOAL_'+state
    if not Path(heartbeat).is_file():return 'PIPELINE_HEARTBEAT_MISSING'
    epoch=load(heartbeat).get('epoch')
    if not isinstance(epoch,(int,float)):return 'PIPELINE_HEARTBEAT_INVALID'
    if now-epoch>stale:return 'PIPELINE_HEARTBEAT_STALE'
    return None
def windows_api():
    if sys.platform!='win32':raise RuntimeError('Windows native Python is required')
    from ctypes import wintypes
    dll=ctypes.WinDLL('kernel32',use_last_error=True)
    func=dll.SetThreadExecutionState
    func.argtypes=[wintypes.DWORD];func.restype=wintypes.DWORD
    def request(flags):
        result=func(flags)
        if not result:raise ctypes.WinError(ctypes.get_last_error())
        return result
    return request
def run(heartbeat,goal,stops,receipt,poll=30,stale=600,api=None,clock=time.time,sleep=time.sleep):
    reason=condition(heartbeat,goal,stops,clock(),stale)
    if reason:
        atomic(receipt,dict(status='NOT_STARTED',reason=reason,epoch=clock(),sleep_inhibition_active=False))
        return reason
    request=api or windows_api()
    request(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)
    try:
        while True:
            reason=condition(heartbeat,goal,stops,clock(),stale)
            if reason:return reason
            atomic(receipt,dict(status='ACTIVE',epoch=clock(),pid=os.getpid(),
                sleep_inhibition_active=True,flags=['ES_CONTINUOUS','ES_SYSTEM_REQUIRED'],
                display_required=False,away_mode_required=False,persistent_power_configuration_changed=False,
                pipeline_heartbeat=str(heartbeat),goal_state=str(goal),stale_timeout_seconds=stale,
                limitation='Does not override explicit user sleep, lid closure, power button, or shutdown.'))
            sleep(poll)
    finally:
        request(ES_CONTINUOUS)
        atomic(receipt,dict(status='CLEARED',epoch=clock(),pid=os.getpid(),
            reason=locals().get('reason') or 'EXCEPTION',sleep_inhibition_active=False,
            persistent_power_configuration_changed=False))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heartbeat',type=Path);p.add_argument('--goal-state',type=Path)
    p.add_argument('--stop-file',type=Path,action='append',default=[])
    p.add_argument('--receipt',type=Path);p.add_argument('--poll',type=int,default=30)
    p.add_argument('--stale',type=int,default=600);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.dry_run:
        print(json.dumps(dict(status='DRY_RUN_NO_REQUEST',flags=ES_CONTINUOUS|ES_SYSTEM_REQUIRED,
            display_required=False,persistent_configuration_changed=False)));return
    if not a.heartbeat or not a.goal_state or not a.receipt:p.error('heartbeat, goal-state and receipt are required')
    if not 5<=a.poll<=60 or not 60<=a.stale<=600:p.error('poll5..60,stale60..600 only')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    print(run(a.heartbeat,a.goal_state,a.stop_file,a.receipt,a.poll,a.stale))
if __name__=='__main__':main()

