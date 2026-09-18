"""One-command detached launcher. Same start command resumes completed checkpoints."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import ROOT, read_json, write_json, validate_config, output_root

JOB=ROOT/'outputs/job.json'
LOG=ROOT/'outputs/training.log'


def process_identity(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text()
        rest=raw[raw.rfind(')')+2:].split()
        return {'start_ticks':rest[19],'state':rest[0]}
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return None


def active_job():
    if not JOB.is_file():
        return None
    info=read_json(JOB);current=process_identity(info['pid'])
    if current and current['start_ticks']==info['start_ticks'] and current['state']!='Z':
        return info
    return None


def show_status():
    job=active_job()
    print('Process:',f'running, PID {job["pid"]}' if job else 'not running')
    latest=ROOT/'outputs/latest.json'
    if latest.is_file():
        folder=ROOT/read_json(latest)['relative_output']
        if (folder/'status.json').is_file():
            print((folder/'status.json').read_text())
        print('Results folder:',folder)
        if (folder/'reports/report.html').is_file():
            print('Report:',folder/'reports/report.html')
        if (folder/'important_outcomes_followup.zip').is_file():
            print('Results ZIP:',folder/'important_outcomes_followup.zip')
    if LOG.is_file():
        with open(LOG,'rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-12000))
            tail=f.read().decode('utf-8',errors='replace').splitlines()[-18:]
        print('\nRecent log:\n'+'\n'.join(tail))
    print('Full log:',LOG)


def start(args):
    existing=active_job()
    if existing:
        print('Already running. No second process started.');show_status();return
    c=read_json(ROOT/'config.json');validate_config(c)
    if args.repair_cuda:
        subprocess.run([sys.executable,str(ROOT/'setup_environment.py'),'--repair-cuda'],check=True,cwd=ROOT)
    elif not args.skip_setup:
        subprocess.run([sys.executable,str(ROOT/'setup_environment.py')],check=True,cwd=ROOT)
    # Unit tests do not touch the internet and run before an unattended job.
    test_log=ROOT/'outputs/prelaunch_tests.log'
    with open(test_log,'w') as stream:
        tested=subprocess.run([sys.executable,'-m','unittest','discover','-p','test_*.py','-q'],
                              stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
    if tested.returncode:
        print(test_log.read_text())
        raise RuntimeError('Prelaunch tests failed. See outputs/prelaunch_tests.log.')
    print('Offline correctness tests passed.',flush=True)
    output=output_root(c)
    if (output/'status.json').is_file() and (read_json(output/'status.json')['stage']=='complete' or (read_json(output/'status.json')['stage']=='evaluation_complete' and not c['run_new_ppo'])):
        print('This exact study is already complete.');show_status();return
    preflight=output/'preflight.json'
    if not preflight.is_file() or not read_json(preflight).get('passed') or args.preflight_again:
        print('Running GPU preflight now; the detached study starts only after it passes.',flush=True)
        subprocess.run([sys.executable,'-u',str(ROOT/'run_study.py'),'--preflight'],check=True,cwd=ROOT)
    with open(LOG,'a',buffering=1) as log:
        log.write('\n=== START OR RESUME '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+' ===\n')
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_study.py')],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'PYTHONUNBUFFERED':'1'})
    current=process_identity(process.pid)
    if not current:
        raise RuntimeError('Worker exited immediately. Inspect outputs/training.log.')
    write_json(JOB,{'pid':process.pid,'start_ticks':current['start_ticks'],
                    'started_unix':time.time(),'python':sys.executable})
    print(f'Started PID {process.pid}. You can close the notebook/browser; keep the RunPod running.',flush=True)
    print('Status: python launch.py status\nPause safely: python launch.py stop\nResume: python launch.py start',flush=True)
    print('Log:',LOG,'\nResults:',output,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','status','stop','report'],nargs='?',default='status')
    parser.add_argument('--skip-setup',action='store_true')
    parser.add_argument('--repair-cuda',action='store_true')
    parser.add_argument('--preflight-again',action='store_true')
    args=parser.parse_args()
    (ROOT/'outputs').mkdir(exist_ok=True)
    if args.action=='status':
        show_status();return
    if args.action=='report':
        if active_job():
            print('The running worker refreshes reports at evaluation boundaries. Opening existing status avoids writing competing reports.')
            show_status();return
        subprocess.run([sys.executable,str(ROOT/'reporting.py')],check=True,cwd=ROOT);return
    if args.action=='stop':
        job=active_job()
        if job:
            os.kill(job['pid'],signal.SIGTERM)
            print('Pause requested. Wait for status=paused before stopping the pod; the current update/batch may take a few minutes.')
        else:
            print('No active study process.')
        return
    with open(ROOT/'outputs/launch.lock','a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another launcher is already preparing this project. Let it finish.')
        start(args)


if __name__=='__main__':
    main()
