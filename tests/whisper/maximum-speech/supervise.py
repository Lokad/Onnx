"""Run a single pinned maximum-speech phase with finite resources and process evidence."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import psutil
from prepare import sha, read, write_new, pin


def alive(pid, birth):
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).create_time() == birth
    except psutil.NoSuchProcess:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--phase', choices=('native', 'managed'), required=True)
    args = parser.parse_args()
    base = args.artifact.resolve()
    frozen = read(base/'frozen.json')
    assert psutil.__version__ == '7.0.0'
    def verify():
        for name, expected in frozen['files'].items():
            assert pin(base/name) == expected, name
    verify()
    phase = args.phase
    if phase == 'managed':
        native = read(base/'native-process.json')
        if not native['complete']:
            recovery = read(base/'native-terminal-recovery.json')
            assert recovery['passed'] and recovery['original_identity'] == native
            assert recovery['native_process_sha256'] == sha(base/'native-process.json')
            assert recovery['native_sha256'] == sha(base/'native/manifest.json')
            assert recovery['native_audit_sha256'] == sha(base/'native-audit.json')
            assert native['error'].endswith('AssertionError: Owned descendant remains\n')
            assert all(not alive(p['pid'],p['create_time']) for p in recovery['terminal_processes'])
        assert native['code'] == 0 and not alive(native['pid'], native['create_time'])
        assert not alive(native['supervisor'], native['supervisor_create_time'])
        audit = read(base/'native-audit.json')
        assert audit['passed'], audit
        result = read(base/'native/manifest.json')['cases'][0]['result']
        assert result['stop_reason'] == 'Completed' and result['processed_seconds'] == 600 and len(result['windows']) >= 20
    output = base/(phase+'-process.json')
    assert not output.exists() and not (base/phase).exists()
    assert psutil.virtual_memory().available >= 15*1024**3, 'Insufficient starting headroom'
    parent = psutil.Process()
    original_affinity = parent.cpu_affinity()
    parent.cpu_affinity([0])
    command = ([sys.executable, '-B', str(base/'native-source/recording/generate_reference.py'), '--models', frozen['models'],
                '--inputs', str(base/'inputs/inputs.json'), '--output', str(base/'native')] if phase == 'native' else
               ['dotnet', str(base/'bin/RecordingReplay.dll'), frozen['models'], str(base/'inputs/inputs.json'),
                str(base/'short/manifest.json'), str(base/'managed')])
    identity = dict(schema=1, phase=phase, command=command, frozen_sha256=sha(base/'frozen.json'),
        supervisor=parent.pid, supervisor_create_time=parent.create_time(), supervisor_affinity=[0],supervisor_sha256=sha(Path(__file__)),
        started=time.time(), host=platform.platform(), complete=False, code=None, members={}, peak_rss=0,
        samples=0, minimum_available_memory=psutil.virtual_memory().available,
        limits=dict(rss=16*1024**3, seconds=7200, available_memory=1024**3))
    def save():
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(identity, indent=2)+'\n', encoding='utf-8')
        temp.replace(output)
    root = Path(__file__).resolve().parents[3]
    if not (root/'eng/campaign_processes.py').exists():
        root = Path.cwd()
    account_path = base/'source/eng/campaign_processes.py'
    spec = importlib.util.spec_from_file_location('process_accounting', account_path)
    account = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(account)
    before = account.snapshot()
    write_new(base/(phase+'-pre.json'), before)
    env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    env.pop('PYTHONPATH', None)
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    child = None
    start = time.monotonic()
    try:
        save()
        with (base/(phase+'.log')).open('x', encoding='utf-8') as log, (base/(phase+'-samples.jsonl')).open('x', encoding='utf-8') as samples:
            parent.cpu_affinity([2])
            try:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            finally:
                parent.cpu_affinity([0])
            process = psutil.Process(child.pid)
            identity.update(pid=child.pid, create_time=process.create_time())
            save()
            while child.poll() is None:
                members = []
                for member in [process]+process.children(recursive=True):
                    try:
                        item = dict(pid=member.pid, create_time=member.create_time(), rss=member.memory_info().rss,
                            affinity=member.cpu_affinity(), cpu_seconds=sum(member.cpu_times()[:2]))
                    except psutil.NoSuchProcess:
                        continue
                    assert item['affinity'] == [2], item
                    if member.pid == process.pid:
                        assert item['create_time'] == identity['create_time']
                    members.append(item)
                    identity['members'][str(member.pid)] = item['create_time']
                available = psutil.virtual_memory().available
                rss = sum(item['rss'] for item in members)
                seconds = time.monotonic()-start
                samples.write(json.dumps(dict(seconds=seconds, rss=rss, available_memory=available, members=members))+'\n')
                samples.flush()
                identity['samples'] += 1
                identity['peak_rss'] = max(identity['peak_rss'], rss)
                identity['minimum_available_memory'] = min(identity['minimum_available_memory'], available)
                identity['seconds'] = seconds
                save()
                assert rss < identity['limits']['rss'] and seconds < identity['limits']['seconds'] and available >= identity['limits']['available_memory'], 'Resource guard'
                time.sleep(.5)
            identity['code'] = child.wait()
            assert identity['code'] == 0, 'Worker failed; preserve original evidence'
            deadline = time.monotonic()+10
            identity['termination_checks'] = []
            while True:
                remaining = [dict(pid=int(pid),create_time=birth) for pid,birth in identity['members'].items() if alive(int(pid),birth)]
                identity['termination_checks'].append(dict(time=time.time(),remaining=remaining))
                save()
                if not remaining:
                    break
                assert time.monotonic() < deadline, 'Owned descendant remains'
                time.sleep(.05)
        verify()
        after = account.snapshot()
        write_new(base/(phase+'-post.json'), after)
        identity['accounting'] = account.foreign_fraction(before, after, parent.pid)
        identity['complete'] = True
    except BaseException:
        identity['error'] = traceback.format_exc()
        raise
    finally:
        if child is not None and child.poll() is None:
            process = psutil.Process(child.pid)
            assert process.create_time() == identity['create_time'], 'Owned worker birth differs'
            for member in process.children(recursive=True):
                assert identity['members'].get(str(member.pid)) == member.create_time(), 'Unverified descendant'
                member.kill()
                member.wait(timeout=10)
            child.kill()
            child.wait()
        if child is not None:
            identity['code'] = child.poll()
        identity.update(ended=time.time(), seconds=time.monotonic()-start)
        save()
        parent.cpu_affinity(original_affinity)
    print(phase, 'complete', identity['seconds'], 'peak', identity['peak_rss'], flush=True)


if __name__ == '__main__':
    main()
