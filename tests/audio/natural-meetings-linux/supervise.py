"""Owned sequential recording workers; observation timeouts never relaunch them."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
import psutil
from common import pin, read, write


def check_sample(sample, limits):
    assert 0 <= sample['seconds'] < limits['seconds'], 'Worker time limit'
    assert sample['available'] >= limits['available'], 'Available memory limit'
    assert sum(m['rss'] for m in sample['members']) < limits['rss'], 'Group RSS limit'
    assert all(m['affinity'] == [2] for m in sample['members']), 'Worker affinity'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    deadline = time.monotonic() + 1
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if os.name != 'nt' or time.monotonic() >= deadline:
                raise
            time.sleep(.01)


def alive(pid, birth):
    try:
        p = psutil.Process(pid)
        return p if p.create_time() == birth else None
    except psutil.NoSuchProcess:
        return None


def verify(base, engine=None):
    frozen = read(base / 'frozen.json')
    for name, wanted in frozen['files'].items():
        assert pin(base / name) == wanted, name
    if engine=='native':
        for name,wanted in frozen['native_files'].items():
            assert pin(Path(name)) == wanted, name
    return pin(base / 'frozen.json')['sha256']


def run(base, engine, family, mode, label=None):
    assert psutil.__version__ == '7.0.0'
    root = base.parents[1]
    out = base / ('process-' + engine + '-' + family + '-' + (label or mode))
    out.mkdir()
    limits = read(base / 'manifest.json')['limits'][engine]
    parent = psutil.Process()
    old_affinity = parent.cpu_affinity()
    parent.cpu_affinity([0])
    result = dict(schema=1, engine=engine, family=family, mode=mode, supervisor=dict(pid=parent.pid,birth=parent.create_time()),
                  complete=False, started=time.time(), limits=limits, members={}, samples=0, peak_rss=0, child=None)
    child = None
    result_code = 2
    try:
        if mode == 'run':
            result['frozen_sha256'] = verify(base,engine)
        assert psutil.virtual_memory().available >= limits['preflight'], 'Insufficient memory before launch'
        assert psutil.disk_usage(str(base)).free >= 8*1024**2, 'Insufficient disk before launch'
        env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
        env.update(PYTHONUTF8='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        if engine == 'native':
            assert sys.platform == 'linux'
            env['PYTHONPATH'] = os.pathsep.join(read(base / 'manifest.json')['python_path'])
            command = [sys.executable,'-X','utf8','-B',str(Path(__file__).with_name('native.py'))]
        else:
            command = ['dotnet', str(base / 'bin/NaturalAsr.dll')]
        command += [str(root), str(base), str(out / 'worker'), family, mode]
        result['command'] = command
        accounting_path = Path(__file__).with_name('campaign_processes.py')
        if not accounting_path.exists(): accounting_path = root / 'eng/campaign_processes.py'
        spec = importlib.util.spec_from_file_location('meeting_accounting',accounting_path)
        account = importlib.util.module_from_spec(spec);spec.loader.exec_module(account)
        before = account.snapshot();write(out/'pre.json',before)
        start = time.monotonic()
        with (out / 'stdout.txt').open('x',encoding='utf-8') as stdout,(out / 'stderr.txt').open('x',encoding='utf-8') as stderr,(out / 'samples.jsonl').open('x',encoding='utf-8') as samples:
            parent.cpu_affinity([2])
            try:
                child = subprocess.Popen(command,cwd=root,env=env,stdout=stdout,stderr=stderr,
                    **(dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=='nt' else dict(start_new_session=True)))
            finally:
                parent.cpu_affinity([0])
            p = psutil.Process(child.pid)
            result['child'] = dict(pid=p.pid,birth=p.create_time())
            while child.poll() is None:
                members = []
                try: group = [p] + p.children(recursive=True)
                except psutil.NoSuchProcess: group = []
                for member in group:
                    try:
                        row = dict(pid=member.pid,birth=member.create_time(),rss=member.memory_info().rss,affinity=member.cpu_affinity())
                        members.append(row)
                        result['members'][str(row['pid'])] = row['birth']
                    except psutil.NoSuchProcess:
                        pass
                sample = dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,members=members)
                samples.write(json.dumps(sample)+'\n');samples.flush()
                result['samples'] += 1
                result['peak_rss'] = max(result['peak_rss'],sum(m['rss'] for m in members))
                save(out/'identity.json',result)
                check_sample(sample,limits)
                time.sleep(.5)
        result['code'] = child.wait()
        result['seconds'] = time.monotonic()-start
        after = account.snapshot();write(out/'post.json',after)
        result['accounting'] = account.foreign_fraction(before,after,parent.pid)
        assert result['code'] == 0 and result['seconds'] < limits['seconds']
        assert read(out/'worker'/('inputs.json' if mode=='inputs' else 'result.json'))
        if mode == 'run':
            assert result['frozen_sha256'] == verify(base,engine)
        result['complete'] = True
        result_code = 0
    except BaseException:
        result['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        # Stop only the process identities created and observed by this supervisor.
        if child is not None:
            owned = dict(result['members'])
            if result['child']: owned[str(result['child']['pid'])] = result['child']['birth']
            for pid,birth in reversed(list(owned.items())):
                process = alive(int(pid),birth)
                if process:
                    try:process.kill()
                    except psutil.NoSuchProcess:pass
            result['code'] = child.wait(timeout=10)
        result['ended'] = time.time()
        save(out/'identity.json',result)
        parent.cpu_affinity(old_affinity)
        write(out/'complete.json',dict(code=result_code))
    return result_code


def campaign(base, engine):
    assert not (base/('campaign-'+engine+'.json')).exists()
    outcomes=[]
    for family in read(base/'manifest.json')['schedule']:
        code=run(base,engine,family,'run')
        outcomes.append(dict(family=family,code=code))
    write(base/('campaign-'+engine+'.json'),dict(complete=True,outcomes=outcomes))
    return 0 if all(r['code']==0 for r in outcomes) else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','launch','campaign'])
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--engine',choices=['native'],required=True)
    p.add_argument('--family',choices=['whisper'])
    p.add_argument('--mode',choices=['inputs','run'],default='run')
    p.add_argument('--label',help='Distinct preserved input-only attempt')
    a=p.parse_args();base=a.artifact.resolve()
    if a.label:assert a.mode=='inputs' and a.label.replace('-','').isalnum()
    if a.action=='run':
        assert a.family is not None
        return run(base,a.engine,a.family,a.mode,a.label)
    assert a.family is None and a.mode=='run' and a.label is None
    if a.action=='campaign':return campaign(base,a.engine)
    assert not (base/('deployment-'+a.engine+'.json')).exists()
    assert not any((base/('process-'+a.engine+'-'+f+'-run')).exists() for f in ['parakeet','whisper'])
    verify(base,a.engine)
    with (base/('supervisor-'+a.engine+'.log')).open('xb') as stream:
        child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'campaign','--artifact',str(base),'--engine',a.engine],
            stdout=stream,stderr=subprocess.STDOUT,**(dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=='nt' else dict(start_new_session=True)))
    value=dict(pid=child.pid,birth=psutil.Process(child.pid).create_time(),started=time.time(),engine=a.engine)
    write(base/('deployment-'+a.engine+'.json'),value);print(json.dumps(value));return 0


if __name__=='__main__':raise SystemExit(main())
