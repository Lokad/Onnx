"""Execute the frozen independent assignments sequentially, preserving all evidence."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import psutil
from contract import manifest, pin, read, LIMITS


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)


def terminal(births):
    for identity in births:
        try:
            assert psutil.Process(identity['pid']).create_time() != identity['birth'], ('owned process live', identity)
        except psutil.NoSuchProcess:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--phase', choices=['aa', 'compare'], required=True)
    parser.add_argument('--gate-sha256')
    args = parser.parse_args()
    base = args.payload.resolve(); meta = manifest(base); phase = args.phase
    full = meta['mode'] == 'full'
    if full:
        assert sys.platform == 'linux'
        terminal(meta['predecessor_births'])
        assert shutil.disk_usage(base).free >= 3*1024**3, 'Need room for complete retained arrays; stage on verified tmpfs'
        if phase == 'compare':
            assert args.gate_sha256 and pin(base/'aa-gate.json')['sha256'] == args.gate_sha256
            gate = read(base/'aa-gate.json')
            assert gate['passed'] is True and gate['timing']['statistical_screen'] is True
            assert gate['frozen'] == pin(base/'frozen.json') and gate['identity'] == pin(base/'result-aa/identity.json')
            for name, wanted in gate['retained_files'].items():
                assert pin(base/name) == wanted
            terminal(gate['resources']['births'])
        else:
            assert args.gate_sha256 is None
        sys.path.insert(0, str(base/'eng'))
        import campaign_processes as accounting
    else:
        assert sys.platform == 'win32' and args.gate_sha256 is None
    out = base/('result-'+phase); out.mkdir(exist_ok=False)
    parent = psutil.Process(); original_affinity = parent.cpu_affinity(); parent.cpu_affinity([0])
    clean = {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    state = dict(phase=phase, mode=meta['mode'], supervisor=dict(pid=parent.pid, birth=parent.create_time()),
                 started=time.time(), complete=False, limits=LIMITS, frozen=pin(base/'frozen.json'), gate_sha256=args.gate_sha256, runs=[])
    save(out/'identity.json', state); started = time.monotonic()
    try:
        for job in meta['schedules'][phase]:
            assert time.monotonic()-started < 30*3600, 'Fixed30hour phase ceiling'
            folder = out/job['name']; folder.mkdir(); child = None
            command = [meta['dotnet']['path'], str(base/'bin/ProcessUncertainty.dll'), meta['model']['path'],
                       str(base/'inputs'/(job['case']+'.json')), str(folder/'output'), job['policy'], job['role'],
                       phase, str(job['cohort']), str(job['case_index']), meta['native']['path'], meta['mode']]
            run = dict(job=job, command=command, started=time.time(), members={}, samples=0, peak_rss=0)
            state['runs'].append(run); save(out/'identity.json', state)
            if full:
                pre = accounting.snapshot(); save(folder/'pre.json', pre)
                (folder/'cpu-before.txt').write_text(Path('/proc/stat').read_text())
            try:
                with (folder/'stdout.txt').open('x') as stdout, (folder/'stderr.txt').open('x') as stderr, (folder/'samples.jsonl').open('x') as samples:
                    flags = {'LOKAD_ONNX_FINGERPRINT_STRINGS': '1', 'LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT': '1'} if phase == 'compare' and job['role'] == 'C' else {}
                    parent.cpu_affinity([2])
                    try:
                        kwargs = dict(start_new_session=True) if full else dict(creationflags=subprocess.CREATE_NO_WINDOW)
                        child = subprocess.Popen(command, cwd=base, env=clean | flags, stdout=stdout, stderr=stderr, **kwargs)
                    finally:
                        parent.cpu_affinity([0])
                    birth = psutil.Process(child.pid).create_time()
                    run['child'] = dict(pid=child.pid, birth=birth); run['members'][str(child.pid)] = birth
                    start = time.monotonic(); save(folder/'identity.json', run)
                    while child.poll() is None:
                        members = []
                        try:
                            owner = psutil.Process(child.pid); assert owner.create_time() == birth
                            for process in [owner]+owner.children(recursive=True):
                                try:
                                    item = dict(pid=process.pid, birth=process.create_time(), rss=process.memory_info().rss, affinity=process.cpu_affinity())
                                    assert item['affinity'] == [2] and item['birth'] >= birth
                                    assert run['members'].get(str(item['pid']), item['birth']) == item['birth']
                                    if full:
                                        threads = []
                                        for thread in process.threads():
                                            try:
                                                affinity = sorted(os.sched_getaffinity(thread.id)); assert affinity == [2]
                                                threads.append(dict(id=thread.id, affinity=affinity))
                                            except ProcessLookupError:
                                                pass
                                        if not threads:
                                            continue  # A process that finished during enumeration is sampled next time if still live.
                                        item['threads'] = threads
                                    run['members'][str(item['pid'])] = item['birth']; members.append(item)
                                except psutil.NoSuchProcess:
                                    pass
                        except psutil.NoSuchProcess:
                            pass
                        sample = dict(seconds=time.monotonic()-start, available=psutil.virtual_memory().available,
                                      disk_free=shutil.disk_usage(base).free, members=members)
                        samples.write(json.dumps(sample)+'\n'); samples.flush()
                        run['samples'] += 1; run['peak_rss'] = max(run['peak_rss'], sum(m['rss'] for m in members))
                        save(folder/'identity.json', run)
                        assert sample['seconds'] < LIMITS['seconds'] and sample['available'] >= LIMITS['available'] and run['peak_rss'] < LIMITS['rss'], 'Resource guard'
                        assert sample['disk_free'] >= 512*1024**2, 'Evidence disk guard'
                        time.sleep(.5)
                    run['code'] = child.wait(); run['seconds'] = time.monotonic()-start
                    assert run['code'] == 0, (job, run['code'])
            finally:
                if child is not None:
                    for pid, actual_birth in reversed(list(run['members'].items())):
                        try:
                            process = psutil.Process(int(pid))
                            if process.create_time() == actual_birth:
                                process.kill()
                        except psutil.NoSuchProcess:
                            pass
                    child.wait(timeout=10)
                    terminal([dict(pid=int(pid), birth=b) for pid, b in run['members'].items()])
                run['ended'] = time.time(); save(folder/'identity.json', run); save(out/'identity.json', state)
            if full:
                post = accounting.snapshot(); save(folder/'post.json', post)
                (folder/'cpu-after.txt').write_text(Path('/proc/stat').read_text())
                run['accounting'] = accounting.foreign_fraction(pre, post, parent.pid)
                save(folder/'identity.json', run); save(out/'identity.json', state)
            print(json.dumps(dict(phase=phase, name=job['name'], seconds=run['seconds'])), flush=True)
        state.update(complete=True, code=0)
    except BaseException:
        state.update(complete=True, code=2, error=traceback.format_exc()); raise
    finally:
        state['ended'] = time.time(); save(out/'identity.json', state); parent.cpu_affinity(original_affinity)


if __name__ == '__main__':
    main()
