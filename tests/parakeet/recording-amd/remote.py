"""Finite Linux recording replay; stdlib only, no inference reference regeneration."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import tarfile
import time
import traceback

RSS_LIMIT = 13 * 1024**3
TIME_LIMIT = 1800
AVAILABLE_MIN = 256 * 1024**2
JOBS = ['managed', 'cli-connected', 'cli-token-limit']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify_file(path, pin):
    assert path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], str(path)


def safe_path(base, name):
    rel = Path(name)
    assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in name, name
    path = (base / rel).resolve()
    assert path.is_relative_to(base.resolve()) and path != base.resolve(), name
    return path


def verify(base):
    for name, pin in read(base / 'bundle.json')['files'].items():
        verify_file(safe_path(base, name), pin)


def install(base):
    """Reconstruct only predeclared duplicate or zero-payload files, then verify."""
    bundle = read(base / 'bundle.json')
    assert not (base / 'installed.json').exists()
    for name, recipe in bundle['recipes'].items():
        path = safe_path(base, name)
        assert not path.exists(), name
        if recipe['kind'] == 'hardlink':
            original = safe_path(base, recipe['source'])
            verify_file(original, bundle['files'][name])
            os.link(original, path)
        else:
            assert recipe['kind'] == 'sparse-zero'
            with path.open('xb') as stream:
                stream.write(bytes.fromhex(recipe['header_hex']))
                stream.truncate(bundle['files'][name]['bytes'])
        verify_file(path, bundle['files'][name])
    verify(base)
    models = Path(bundle['models'])
    for name, pin in read(base / 'reference/assets.json')['files'].items():
        verify_file(models / name, pin)
    write_new(base / 'installed.json', dict(installed_at=time.time(), bundle_sha256=sha(base/'bundle.json'),
        available_bytes=os.statvfs(base).f_bavail * os.statvfs(base).f_frsize))
    print('Installed and verified every logical file and all six model assets.', flush=True)


def proc(pid):
    try:
        line = Path('/proc', str(pid), 'stat').read_text()
        fields = line[line.rfind(')')+2:].split()
        status = dict(v.split(':', 1) for v in Path('/proc', str(pid), 'status').read_text().splitlines() if ':' in v)
        return dict(pid=int(pid), start=int(fields[19]), group=int(fields[2]), state=fields[0],
            rss=int(status.get('VmRSS', '0 kB').split()[0])*1024,
            affinity=status['Cpus_allowed_list'].strip(),
            cpu_seconds=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def members(group):
    result = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            item = proc(path.name)
            if item and item['group'] == group and item['state'] != 'Z':
                result.append(item)
    return result


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def check_sample(sample, group, birth):
    assert 0 <= sample['seconds'] < TIME_LIMIT, 'Time guard'
    assert sample['available_memory'] >= AVAILABLE_MIN, 'Available memory guard'
    seen = set()
    for member in sample['members']:
        assert member['group'] == group and member['start'] >= birth and member['affinity'] == '2', 'Process identity/affinity'
        assert member['rss'] >= 0 and member['cpu_seconds'] >= 0 and member['pid'] not in seen, 'Process accounting'
        if member['pid'] == group:
            assert member['start'] == birth, 'Root process birth'
        seen.add(member['pid'])
    assert sum(m['rss'] for m in sample['members']) < RSS_LIMIT, 'RSS guard'


def stop(child, birth):
    root = proc(child.pid)
    if child.poll() is None:
        assert root and root['start'] == birth, 'Cannot verify owned root before stopping'
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    # These dotnet jobs do not spawn detached children; retain and reject any survivors.
    assert not members(child.pid), 'Worker process group remains'
    return child.wait()


def run(base):
    verify(base)
    assert read(base/'installed.json')['bundle_sha256'] == sha(base/'bundle.json')
    out = base/'result'
    out.mkdir()
    os.sched_setaffinity(0, {0})
    parent = proc(os.getpid())
    identity = dict(schema=1, supervisor=parent, started=time.time(), complete=False, runs=[],
        bundle_sha256=sha(base/'bundle.json'), limits=dict(rss=RSS_LIMIT, seconds=TIME_LIMIT, available_memory=AVAILABLE_MIN))
    def save():
        temp = out/'identity.tmp'
        temp.write_text(json.dumps(identity, indent=2)+'\n', encoding='utf-8')
        temp.replace(out/'identity.json')
    clean = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    bundle = read(base/'bundle.json')
    models = bundle['models']
    jobs = [(JOBS[0], ['dotnet', str(base/'bin/ParakeetRecordingReplay.dll'), models,
                      str(base/'inputs/inputs.json'), str(out/'managed')])]
    for name, extra in [(JOBS[1], []), (JOBS[2], ['--max-tokens=2'])]:
        jobs.append((name, ['dotnet', str(base/'bin/Lokad.Onnx.CLI.dll'), 'transcribe', models,
            str(base/'inputs/connected.wav'), '--model-type=parakeet', '--recording', '--json']+extra))
    code = 2
    try:
        save()
        (out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
        (out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet', '--info'], text=True))
        for name, command in jobs:
            verify(base)
            row = dict(name=name, command=command, started=time.time(), code=None, samples=0, peak_rss=0, members={})
            start = time.monotonic()
            child = None
            birth = None
            with (out/(name+'.stdout')).open('x') as stdout, (out/(name+'.stderr')).open('x') as stderr, (out/(name+'-samples.jsonl')).open('x') as samples:
                os.sched_setaffinity(0, {2})
                try:
                    child = subprocess.Popen(command, cwd=base, env=clean, stdout=stdout, stderr=stderr, start_new_session=True)
                finally:
                    os.sched_setaffinity(0, {0})
                first = proc(child.pid)
                assert first, 'No initial process identity'
                birth = first['start']
                row.update(pid=child.pid, start=birth)
                identity['runs'].append(row)
                save()
                try:
                    while child.poll() is None:
                        sample = dict(seconds=time.monotonic()-start, members=members(child.pid), available_memory=available())
                        samples.write(json.dumps(sample)+'\n')
                        samples.flush()
                        row['samples'] += 1
                        row['peak_rss'] = max(row['peak_rss'], sum(m['rss'] for m in sample['members']))
                        for member in sample['members']:
                            row['members'][str(member['pid'])] = member['start']
                        save()
                        check_sample(sample, child.pid, birth)
                        time.sleep(.5)
                finally:
                    row['code'] = stop(child, birth)
                    row.update(seconds=time.monotonic()-start, ended=time.time())
                    save()
            assert row['code'] == 0 and row['seconds'] < TIME_LIMIT, name
            verify(base)
            print(name, 'complete', row['seconds'], row['peak_rss'], flush=True)
        identity['complete'] = True
        code = 0
    except BaseException:
        identity['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        identity['ended'] = time.time()
        save()
        (base/'complete.txt').write_text(str(code)+'\n')
    return code


def launch(base):
    verify(base)
    assert not (base/'deployment.json').exists() and not (base/'result').exists()
    with (base/'supervisor.log').open('x') as log:
        child = subprocess.Popen(['taskset', '-c', '0', 'python3', '-B', '-u', str(base/'remote.py'), 'run', str(base)],
            cwd=base, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = proc(child.pid)
    assert identity
    deployment = dict(pid=child.pid, start=identity['start'], started=time.time(), bundle_sha256=sha(base/'bundle.json'))
    write_new(base/'deployment.json', deployment)
    print(json.dumps(deployment))


def collect(base):
    deployment = read(base/'deployment.json')
    identity = read(base/'result/identity.json')
    assert deployment['pid'] == identity['supervisor']['pid'] and deployment['start'] == identity['supervisor']['start']
    roots = [dict(pid=deployment['pid'], start=deployment['start'])]
    roots += [dict(pid=int(pid), start=birth) for row in identity['runs'] for pid,birth in row['members'].items()]
    roots += [dict(pid=row['pid'], start=row['start']) for row in identity['runs']]
    for item in roots:
        live = proc(item['pid'])
        assert live is None or live['start'] != item['start'], 'Observed process still present'
    for group in [deployment['pid']] + [r['pid'] for r in identity['runs']]:
        assert not members(group), 'Observed group still present'
    verify(base)
    # Reusable inputs and binaries are retained locally and bound by bundle.json.
    paths = [p for p in sorted(base.rglob('*')) if p.is_file() and p.relative_to(base).parts[0] not in ('bin', 'inputs')]
    assert not (base/'collection.json').exists()
    inventory = {p.relative_to(base).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths}
    receipt = dict(schema=1, collected_at=time.time(), terminal_processes=roots, files=inventory,
        complete=identity['complete'], code=int((base/'complete.txt').read_text()),
        checkout=subprocess.check_output(['git', '-C', '/home/vermorel/Onnx', 'rev-parse', 'HEAD'], text=True).strip())
    write_new(base/'collection.json', receipt)
    archive = base.with_name(base.name+'-results.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        for p in paths + [base/'collection.json']:
            tar.add(p, arcname=p.relative_to(base).as_posix(), recursive=False)
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), collection_sha256=sha(base/'collection.json'))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('install', 'launch', 'run', 'collect'))
    p.add_argument('base', type=Path)
    a = p.parse_args()
    return globals()[a.action](a.base.resolve()) or 0


if __name__ == '__main__':
    raise SystemExit(main())
