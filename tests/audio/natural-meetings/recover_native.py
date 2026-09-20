"""One recorded retry of a terminal, memory-stopped native Whisper worker.

All original files, executable inputs, limits and recording options are retained.
An additional sixty-second stable-memory preflight precedes the sole retry.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import psutil
from common import load,pin,read,write


DESTINATION='process-native-whisper-recovery1'


def verify(base):
    plan=read(base/'recovery-plan.json')
    assert plan['destination']==DESTINATION and plan['engine']=='native' and plan['family']=='whisper'
    assert plan['frozen']==pin(base/'frozen.json') and plan['limits']==read(base/'manifest.json')['limits']['native']
    for name,wanted in dict(plan['sources'],**plan['failed_files']).items():assert pin(base/name)==wanted,name
    for item in plan['terminal_processes']:
        try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
        except psutil.NoSuchProcess:pass
    return plan


def prepare(base):
    assert os.name=='nt' and not (base/'recovery-plan.json').exists() and not (base/DESTINATION).exists()
    root=base.parents[1]
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    directory=base/'process-native-whisper-run';state=read(directory/'identity.json')
    assert state['complete'] is False and read(directory/'complete.json')==dict(code=2)
    assert 'Available memory limit' in state['error'] and state['child'] is not None
    assert {p.name for p in (directory/'worker').iterdir()}=={'00.json'}
    assert read(base/'campaign-native.json')==dict(complete=True,outcomes=[dict(family='parakeet',code=0),dict(family='whisper',code=2)])
    births=dict((int(p),b) for p,b in state['members'].items())
    for item in [state['supervisor'],state['child'],read(base/'deployment-native.json')]:births[item['pid']]=item['birth']
    for pid,birth in births.items():
        try:assert psutil.Process(pid).create_time()!=birth,(pid,birth)
        except psutil.NoSuchProcess:pass
    snapshot=base/'recovery-source';snapshot.mkdir()
    for name in ['recover_native.py','common.py']:shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    plan=dict(schema=1,created=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        engine='native',family='whisper',destination=DESTINATION,frozen=pin(base/'frozen.json'),limits=state['limits'],
        cases=[c['name'] for c in read(base/'manifest.json')['cases']],stable_preflight_seconds=60,preflight_timeout_seconds=900,
        sources={p.relative_to(base).as_posix():pin(p) for p in snapshot.iterdir()},
        failed_files={p.relative_to(base).as_posix():pin(p) for p in directory.rglob('*') if p.is_file()},
        terminal_processes=[dict(pid=p,birth=b) for p,b in sorted(births.items())],
        reason='Original worker was stopped by the fixed available-memory guard. Preserve it as failed; retry the entire three-call ownership sequence once without changing numerical or resource limits.',
        repeat_policy='Compare the earlier completed ES2004a public result against the retry and retain every difference.')
    write(base/'recovery-plan.json',plan);verify(base);print('Prepared one native Whisper retry; original failed files remain immutable.')


def run(base):
    plan=verify(base);assert not (base/DESTINATION).exists() and not (base/'recovery-outcome.json').exists()
    supervisor=psutil.Process();supervisor.cpu_affinity([0])
    identity=dict(pid=supervisor.pid,birth=supervisor.create_time())
    write(base/'recovery-supervisor.json',identity)
    start=time.monotonic();stable=None;ready=False
    with (base/'recovery-preflight.jsonl').open('x',encoding='utf-8') as stream:
        while time.monotonic()-start<plan['preflight_timeout_seconds']:
            now=time.monotonic();available=psutil.virtual_memory().available
            if available<plan['limits']['preflight']:stable=None
            elif stable is None:stable=now
            seconds=0 if stable is None else now-stable
            stream.write(json.dumps(dict(elapsed=now-start,available=available,stable_seconds=seconds))+'\n');stream.flush()
            if seconds>=plan['stable_preflight_seconds']:ready=True;break
            time.sleep(2)
    if not ready:
        write(base/'recovery-outcome.json',dict(code=2,child_created=False,reason='Stable preflight not reached',supervisor=identity));return 2
    verify(base)
    # Execute the original frozen supervisor and native runner with a new output name.
    frozen=load('frozen_natural_asr_supervisor',base/'runtime/supervise.py')
    code=frozen.run(base,'native','whisper','run','recovery1')
    write(base/'recovery-outcome.json',dict(code=code,child_created=read(base/DESTINATION/'identity.json')['child'] is not None,supervisor=identity))
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','launch','run']);p.add_argument('--artifact',type=Path,required=True)
    a=p.parse_args();base=a.artifact.resolve()
    if a.action=='prepare':prepare(base);return 0
    plan=verify(base)
    assert pin(Path(__file__))==plan['sources']['recovery-source/recover_native.py']
    if a.action=='run':return run(base)
    assert not (base/'recovery-deployment.json').exists() and not (base/DESTINATION).exists()
    with (base/'recovery-supervisor.log').open('xb') as stream:
        child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run','--artifact',str(base)],
            stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    value=dict(pid=child.pid,birth=psutil.Process(child.pid).create_time(),created=time.time())
    write(base/'recovery-deployment.json',value);print(json.dumps(value));return 0


if __name__=='__main__':raise SystemExit(main())
