"""Verify completed functional smokes and exercise damaged evidence refusals."""
from pathlib import Path
import argparse
import array
import copy
import json
import math
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil
from worker_audit import pin, read, write, worker, telemetry
from protocol import CORE, schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    args = parser.parse_args()
    base = args.artifact.resolve()
    assert not (base/'local-verification.json').exists()
    audit = read(base/'smoke-audit.json')
    assert audit['passed'] is True and len(audit['jobs']) == 60
    state = read(base/'smoke-process/identity.json')
    assert state['complete'] is True and state['code'] == 0
    for name, expected in audit['binaries'].items():
        assert pin(base/'bin'/name) == expected, name
    assert audit['binaries']['Lokad.Onnx.dll']['sha256'] == CORE
    for name, expected in audit['source'].items():
        assert pin(Path(__file__).with_name(name)) == expected, name
    expected_jobs = []
    for phase in ('aa', 'compare'):
        for source in schedule(phase):
            if source['cohort'] == 0:
                expected_jobs.append(source | dict(name=phase+'-'+source['name'], phase=phase))
    assert audit['jobs'] == expected_jobs
    samples = {j['name']: [json.loads(line) for line in (base/'smoke-process'/j['name']/'samples.jsonl').read_text().splitlines()] for j in expected_jobs}
    assert telemetry(state, samples, expected_jobs) == audit['resources']
    for birth in audit['resources']['births']:
        try:
            assert psutil.Process(birth['pid']).create_time() != birth['birth'], ('live process', birth)
        except psutil.NoSuchProcess:
            pass
    assert sys.byteorder == 'little'
    scalar_values, conditioning, measured = 0, 0, 0
    rows = []
    def check(folder, job):
        return worker(folder, base/'inputs', audit['model'], audit['binaries']['ProcessUncertainty.dll'],
                      audit['binaries']['Microsoft.ML.OnnxRuntime.dll'], audit['native'], job, job['phase'], True)
    for job in expected_jobs:
        folder = base/'smoke-process'/job['name']/'output'
        value = check(folder, job)
        fixture = read(base/'inputs'/(job['case']+'.json'))
        reference = array.array('f', (base/'inputs'/fixture['reference_file']).read_bytes())
        for stage in ('before', 'after'):
            actual = array.array('f', (folder/(stage+'.f32')).read_bytes())
            assert len(actual) == len(reference)
            error = max(abs(float(x)-float(y))/max(1, abs(float(y))) for x, y in zip(actual, reference, strict=True))
            assert math.isfinite(error) and error <= 1e-4 and abs(error-value[stage+'_error']) <= 1e-15
            scalar_values += len(actual)
        conditioning += len(value['conditioning'])
        measured += len(value['measured'])
        rows.append(dict(job=job, error=value['after_error'], hash=value['output_sha256']))
    for index in range(5):
        for native in (False, True):
            assert len({r['hash'] for r in rows if r['job']['case_index'] == index and (r['job']['role'] == 'N') == native}) == 1

    job = next(j for j in expected_jobs if j['phase'] == 'compare' and j['role'] == 'C' and j['case_index'] == 0)
    original = base/'smoke-process'/job['name']/'output'
    changes = [dict(passed=False), dict(core_sha256='0'*64), dict(probe_sha256='0'*64),
               dict(wide_enabled=False), dict(enabled=False), dict(flags={}), dict(affinity=1),
               dict(processor_count=2), dict(runtime='10.0.8'), dict(unchanged_inputs=False),
               dict(unchanged_held_output=False), dict(cache_entries=0), dict(output_sha256='0'*64),
               dict(input_sha256='0'*64), dict(reference_sha256='0'*64), dict(after_error=1),
               dict(conditioned=0), dict(conditioning_wall_ticks=0), dict(measured=[])]
    refusals = 0
    with tempfile.TemporaryDirectory(prefix='damaged-', dir=base) as temporary:
        damaged = Path(temporary)/'output'
        shutil.copytree(original, damaged)
        value = read(original/'result.json')
        for change in changes:
            (damaged/'result.json').write_text(json.dumps(value | change), encoding='utf8')
            try:
                check(damaged, job)
            except (AssertionError, ValueError, KeyError, TypeError):
                refusals += 1
            else:
                raise AssertionError(('damaged worker accepted', change))
        for field in ('execute', 'request', 'bytes', 'g0'):
            v = copy.deepcopy(value)
            v['measured'][0][field] = -1
            (damaged/'result.json').write_text(json.dumps(v), encoding='utf8')
            try:
                check(damaged, job)
            except AssertionError:
                refusals += 1
            else:
                raise AssertionError(('damaged row accepted', field))
        (damaged/'result.json').write_text(json.dumps(value), encoding='utf8')
        binary = bytearray((damaged/'after.f32').read_bytes())
        binary[:4] = b'\0\0\xc0\x7f'  # Quiet NaN.
        (damaged/'after.f32').write_bytes(binary)
        try:
            check(damaged, job)
        except AssertionError:
            refusals += 1
        else:
            raise AssertionError('damaged full output accepted')
    assert refusals == 24
    changed = subprocess.check_output(['git', 'diff', '--name-only', '4f10e8b', 'HEAD', '--', 'src/Lokad.Onnx'], cwd=ROOT, text=True).splitlines()
    assert changed == ['src/Lokad.Onnx/ComputationalGraph.cs']
    diff = subprocess.check_output(['git', 'diff', '4f10e8b', 'HEAD', '--', changed[0]], cwd=ROOT, text=True)
    (base/'core-source-bridge.diff').write_text(diff, encoding='utf8')
    added = [line for line in diff.splitlines() if line.startswith('+') and not line.startswith('+++')]
    removed = [line for line in diff.splitlines() if line.startswith('-') and not line.startswith('---')]
    assert len(added) == 11 and not removed
    assert any('CreateExecution(ExecutionOptions? options, long maximumReleasedBufferBytes)' in line for line in added)
    pins = {p.relative_to(base).as_posix(): pin(p) for p in sorted(base.rglob('*')) if p.is_file() and 'obj' not in p.relative_to(base).parts}
    write(base/'local-verification.json', dict(passed=True, workers=60, measured_calls=measured,
          conditioning_calls=conditioning, independent_scalar_output_values=scalar_values,
          damaged_record_refusals=refusals, terminal_births=len(audit['resources']['births']),
          core_source_bridge=dict(changed=changed, added_lines=11, removed_lines=0), pins=pins))
    print(json.dumps(dict(passed=True, workers=60, measured=measured, conditioning=conditioning,
                          scalar_values=scalar_values, refusals=refusals, pins=len(pins))))


if __name__ == '__main__':
    main()
