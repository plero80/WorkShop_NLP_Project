"""Start/status/pause a detached best-of-N worker; never alter old experiments."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from bon_io import HERE, read, write

def latest(project):
    p=Path(read(project/'best_of_n_outputs/latest.json')['output'])
    return p if p.exists() else project/'best_of_n_outputs'/p.name

def alive(pidfile):
    if not pidfile.exists():return False
    try:
        obj=read(pidfile)
        raw=Path(f'/proc/{obj["pid"]}/cmdline').read_bytes()
        return b'bon_run.py' in raw
    except (OSError,KeyError,ValueError):return False

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','status','pause'])
    p.add_argument('--project',type=Path,required=True);p.add_argument('--oracle',type=Path)
    p.add_argument('--settings',type=Path,default=HERE/'settings.json')
    p.add_argument('--phase',choices=['development','confirmation'],default='development');a=p.parse_args()
    project=a.project.resolve();base=project/'best_of_n_outputs';base.mkdir(exist_ok=True)
    pidfile=base/'worker.pid'
    if a.action=='start':
        import fcntl
        starter=open(base/'launcher.lock','a+')
        try:fcntl.flock(starter,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another launch is already in progress.')
        if alive(pidfile):raise SystemExit('A best-of-N worker is already running. Use status.')
        # Only clear the previous best-of-N pause flag; original study controls are untouched.
        if (base/'latest.json').exists():
            old=latest(project)
            if (old/'PAUSE').exists():(old/'PAUSE').unlink()
        log=base/f'{a.phase}.log'
        cmd=[sys.executable,'-u',str(HERE/'bon_run.py'),'--project',str(project),'--settings',str(a.settings.resolve()),'--phase',a.phase]
        if a.oracle:cmd+=['--oracle',str(a.oracle.resolve())]
        with open(log,'ab',buffering=0) as f:
            worker=subprocess.Popen(cmd,cwd=project,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write(pidfile,{'pid':worker.pid,'phase':a.phase,'started_unix':time.time(),'log':str(log)})
        time.sleep(.5)
        if worker.poll() is not None:raise SystemExit('Worker exited. Inspect '+str(log))
        print('Worker started:',worker.pid,'\nLog:',log,'\nPreflight runs first; inspect status for success or errors. Keep the pod on.')
    elif a.action=='pause':
        if alive(pidfile):
            import signal
            os.kill(read(pidfile)['pid'],signal.SIGTERM)
            print('Pause requested. Wait for paused status before shutting down.')
        else:print('No live best-of-N worker.')
    else:
        print('Worker alive:',alive(pidfile))
        if (base/'latest.json').exists():
            out=latest(project);print('Output:',out)
            if (out/'status.json').exists():print(json.dumps(read(out/'status.json'),indent=2))
        if pidfile.exists():
            log=Path(read(pidfile)['log'])
            if log.exists():
                with open(log,'rb') as f:f.seek(max(0,log.stat().st_size-4500));print(f.read().decode(errors='replace'))
if __name__=='__main__':main()
