"""Prove both assembly identity guards reject bad arguments before inference."""
import os
from pathlib import Path
import subprocess
import sys
import time
from protocol import pin, read, save

DOTNET = '/home/vermorel/.dotnet/dotnet'
CASES = [(role, guard) for role in ['selected', 'candidate'] for guard in ['core', 'data']]


def command_for(spec, role, guard):
    assert (role, guard) in CASES
    hashes = [spec['identities'][role][name]['sha256'] for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']]
    hashes[0 if guard == 'core' else 1] = '0' * 64
    return [DOTNET, f'runtimes/{role}/GraphQualification.dll', 'assets',
            f'manifests/{role}-pyannote.json', f'identity-probes/{role}-{guard}-output', *hashes]


def review(base, spec):
    built = read(base/'built.json')
    rows = read(base/'identity-probes/probes.json')
    assert [(r['role'], r['guard']) for r in rows] == CASES
    assert len({(r['child']['pid'], r['child']['birth']) for r in rows}) == 4
    for row in rows:
        role, guard = row['role'], row['guard']
        assert row['command'] == command_for(spec, role, guard)
        assert row['consumer'] == built['consumer'] == pin(base/f'runtimes/{role}/GraphQualification.dll')
        for name, wanted in spec['identities'][role].items():
            assert pin(base/'runtimes'/role/name) == wanted
        assert row['child']['pid'] > 0 and row['child']['birth'] > 0
        assert row['terminal'] and isinstance(row['code'], int) and row['code'] != 0
        assert not row['timed_out'] and 0 <= row['seconds'] < 60
        assert row['affinity'] == [2] and not row['output_created']
        assert not (base/row['command'][4]).exists()
        for stream in ['stdout', 'stderr']:
            path = base/f'identity-probes/{role}-{guard}.{stream}'
            assert pin(path) == row[stream]
        error = (base/f'identity-probes/{role}-{guard}.stderr').read_text(encoding='utf8')
        assert error.splitlines()[0] == 'Unhandled exception. System.IO.InvalidDataException: Qualified ' + guard
        assert 'Qualified ' + ('data' if guard == 'core' else 'core') not in error
    return dict(passed=True,probes=4,guards=['core','data'],roles=['selected','candidate'],
                rejection_before_output=True,consumer=built['consumer'])


def main():
    import psutil
    import resource
    assert sys.platform == 'linux' and not sys.flags.optimize
    base = Path(__file__).resolve().parents[1]
    assert os.sched_getaffinity(0) == {2}
    # The four expected unhandled exceptions must not produce core dumps.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    spec, built = read(base/'payload.json'), read(base/'built.json')
    target = base/'identity-probes/probes.json'
    assert not target.exists()
    rows = []
    for role, guard in CASES:
        command = command_for(spec, role, guard)
        assert not (base/command[4]).exists()
        row = dict(role=role,guard=guard,command=command,consumer=built['consumer'],
                   terminal=False,timed_out=False)
        rows.append(row)
        started = time.monotonic()
        with (base/f'identity-probes/{role}-{guard}.stdout').open('x') as out, \
             (base/f'identity-probes/{role}-{guard}.stderr').open('x') as err:
            child = subprocess.Popen(command,cwd=base,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            try:
                process = psutil.Process(child.pid)
                row.update(child=dict(pid=child.pid,birth=process.create_time()),affinity=process.cpu_affinity())
                save(target,rows)
                child.wait(timeout=55)
            except subprocess.TimeoutExpired:
                row['timed_out'] = True
                raise
            finally:
                if child.poll() is None:
                    child.kill()
                row.update(code=child.wait(timeout=5),terminal=True,seconds=time.monotonic()-started,
                           output_created=(base/command[4]).exists())
                save(target,rows)
        for stream in ['stdout','stderr']:
            row[stream] = pin(base/f'identity-probes/{role}-{guard}.{stream}')
        save(target,rows)
    print(review(base,spec))


if __name__ == '__main__':
    main()
