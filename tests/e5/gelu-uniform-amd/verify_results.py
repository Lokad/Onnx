"""Independently check retained records, terminal births and source/data identities.

Run once after audit.py. This never runs inference or changes a remote artifact.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import time

from audit import inspect_timing
from code_audit import inspect as inspect_code
from prepare import CASES, pin, read, write


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def files_equal(base, inventory):
    for name, wanted in inventory.items():
        assert pin(base / name) == wanted, name


def must_refuse(action):
    try:
        action()
    except (AssertionError, KeyError, ValueError):
        return
    raise AssertionError('Damaged record was accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    args = parser.parse_args()
    base = args.artifact.resolve()
    root = Path(__file__).resolve().parents[3]
    assert base == root / 'artifacts/gelu-uniform-amd-20260920'
    assert not (base / 'verification.json').exists() and not (base / 'closed.json').exists()
    collected = base / 'collected'
    payload = base / 'payload'
    collection = read(collected / 'collection.json')
    assert collection['passed']
    files_equal(collected, collection['files'])
    assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()} == set(collection['files']) | {'collection.json'}
    bundle = read(payload / 'bundle.json')
    assert pin(payload / 'bundle.json') == pin(collected / 'bundle.json')
    files_equal(payload, bundle['files'])
    assert collection['verified_reusable_files'] == {n: v for n, v in bundle['files'].items() if n.split('/')[0] in ('bin', 'data')}
    assert {p.relative_to(payload).as_posix() for p in payload.rglob('*') if p.is_file()} == set(bundle['files']) | {'bundle.json'}
    census = root / 'artifacts/gelu-branch-census-20260920'
    assert pin(census / 'closed.json') == bundle['census_receipt']
    files_equal(census, read(census / 'closed.json')['files'])
    for name in CASES:
        for kind, key in [('x', 'input_sha256'), ('bias', 'bias_sha256'), ('y', 'output_sha256')]:
            digest = hashlib.sha256()
            for layer in range(12):
                digest.update((census / f'capture/{name}/{layer:02d}-{kind}.f32').read_bytes())
            assert digest.hexdigest() == bundle['banks'][name][key]

    report = read(base / 'audit.json')
    accounting = module('frozen_accounting', payload / 'campaign_processes.py')
    births = set()
    samples_total = 0
    for phase in ['code', 'result']:
        state = read(collected / phase / 'identity.json')
        assert state['complete'] and not state.get('error')
        assert (collected / phase / 'complete.txt').read_text().strip() == '0'
        supervisor = state['supervisor']
        births.add((supervisor['pid'], supervisor['start']))
        assert supervisor['affinity'] == '0'
        for run in state['runs']:
            assert run['code'] == 0 and not (collected / phase / (run['name'] + '.stderr')).read_bytes()
            births.add((run['pid'], run['start']))
            births.update((int(pid), start) for pid, start in run['members'].items())
            before, after = [read(collected / phase / (run['name'] + '-' + suffix + '.json')) for suffix in ['pre', 'post']]
            assert accounting.foreign_fraction(before, after, supervisor['pid']) == run['accounting']
            samples_total += run['samples']
            proof = read(collected / phase / (run['name'] + '.json'))
            assert proof['passed'] and proof['width'] == 8 and proof['fma'] and proof['avx512']
            assert (proof['cases'], proof['compared']) == (1575, 102364884)
            assert [(c['name'], c['layer']) for c in proof['captures']] == [(c, i) for c in CASES for i in range(12)]
            for c in proof['captures']:
                assert c['values'] * 4 == bundle['files'][f"data/capture/{c['name']}/{c['layer']:02d}-x.f32"]['bytes']
    assert births == {(v['pid'], v['start']) for v in collection['terminal_processes']}
    assert samples_total == sum(r['samples'] for r in report['resources'])

    gate = read(collected / 'code-gate.json')
    assembly = (collected / 'code/jit.txt').read_text()
    code = inspect_code(assembly)
    assert all(gate[key] == value for key, value in code.items())
    assert gate['code_auditor_sha256'] == pin(Path(__file__).with_name('code_audit.py'))['sha256'] == pin(collected / 'code_audit.py')['sha256']
    branch = re.search(r'\bje\s+(?:SHORT\s+)?' + gate['fast_block'] + r'\b', assembly)[0]
    join = re.search(r'\bjmp\s+(?:SHORT\s+)?' + gate['join_block'] + r'\b', code['skipped_path'])[0]
    for damaged in [assembly.replace(branch, branch.replace('je', 'jne', 1), 1),
                    assembly.replace(branch, branch.replace(gate['fast_block'], gate['branch_block']), 1),
                    assembly.replace(join, join.replace(gate['join_block'], gate['branch_block']), 1)]:
        assert damaged != assembly
        must_refuse(lambda: inspect_code(damaged))

    totals = dict(measured_batches=0, warmup_batches=0, bank_calls=0, warmup_bank_calls=0, allocated=0, gc=[0, 0, 0])
    means = {}
    for visit in range(4):
        value = read(collected / f'result/{visit}.json.timing.json')
        assert len(inspect_timing(value, visit)) == 20
        for c in value['results']:
            for label, count_key, bank_key in [('measured', 'measured_batches', 'bank_calls'), ('warmup', 'warmup_batches', 'warmup_bank_calls')]:
                totals[count_key] += len(c[label])
                totals[bank_key] += sum(s['repeats'] for s in c[label])
            for variant in ['Product', 'CopyA', 'CopyB', 'Conditional']:
                observations = [s for s in c['measured'] if s['variant'] == variant]
                assert len(observations) == 48 and len({s['repeats'] for s in observations}) == 1
                # Independent integer total rather than the auditor's mean of float sample times.
                mean = sum(s['ticks'] for s in observations) * 1000 / (sum(s['repeats'] for s in observations) * value['frequency'])
                means[c['name'], visit, variant] = mean
                row, = [r for r in report['rows'] if (r['case'], r['visit'], r['variant']) == (c['name'], visit, variant)]
                assert math.isclose(mean, row['mean_ms'], rel_tol=1e-14)
                totals['allocated'] += sum(s['allocated'] for s in observations)
                for i in range(3):
                    totals['gc'][i] += sum(s['gc'][i] for s in observations)
    assert (totals['measured_batches'], totals['warmup_batches'], totals['bank_calls'], totals['warmup_bank_calls']) == (3840, 1280, 87552, 29184)
    controls, candidate = True, True
    for c in report['cases']:
        name = c['name']
        m = {v: math.fsum(means[name, i, v] for i in range(4)) / 4 for v in ['Product', 'CopyA', 'CopyB', 'Conditional']}
        for v in m:
            assert math.isclose(m[v], c['means_ms'][v], rel_tol=1e-14)
        control = 1 / 1.01 <= m['CopyB'] / m['CopyA'] <= 1.01 and all(1 / 1.02 <= means[name, i, 'CopyB'] / means[name, i, 'CopyA'] <= 1.02 for i in range(4))
        gain = name not in CASES[1:4] or all(m['Conditional'] <= .98 * m[v] for v in ['Product', 'CopyA', 'CopyB'])
        no_regression = all(m['Conditional'] <= 1.01 * m[v] for v in ['Product', 'CopyA', 'CopyB']) and all(means[name, i, 'Conditional'] <= 1.02 * means[name, i, v] for i in range(4) for v in ['Product', 'CopyA', 'CopyB'])
        assert (control, gain, no_regression) == (c['controls_passed'], c['gain_passed'], c['no_regression_passed'])
        controls &= control
        candidate &= gain and no_regression
    assert (controls, candidate, controls and candidate) == (report['controls_passed'], report['candidate_screen_passed'], report['overall_passed'])

    actual = read(collected / 'result/0.json.timing.json')
    changes = [lambda v: v['results'].pop(), lambda v: v['results'][0]['measured'].pop(),
               lambda v: v['results'][0]['measured'][0].update(repeats=63),
               lambda v: v['results'][0]['measured'][0].update(variant='Conditional'),
               lambda v: v['results'][0]['measured'][0].update(output_sha256='0' * 64),
               lambda v: v['results'][0]['measured'][0].update(ticks=0),
               lambda v: v.update(flags=['DOTNET_TieredCompilation']),
               lambda v: v.update(case_order=list(reversed(CASES)))]
    for change in changes:
        damaged = copy.deepcopy(actual)
        change(damaged)
        must_refuse(lambda: inspect_timing(damaged, 0))
    altered = copy.deepcopy(bundle['files'])
    altered['bin/Probe.dll']['sha256'] = '0' * 64
    must_refuse(lambda: files_equal(payload, altered))

    # Fresh read-only /proc checks; never infer termination from a timestamp or saved status.
    script = """from pathlib import Path
import json
births = BIRTHS
checked = []
for pid, expected in births:
    path = Path('/proc') / str(pid) / 'stat'
    try:
        text = path.read_text(); current = int(text[text.rfind(')') + 1:].split()[19])
    except (FileNotFoundError, ProcessLookupError): current = None
    assert current != expected, (pid, expected)
    checked.append(dict(pid=pid, start=expected, observed_start=current))
print(json.dumps(checked))
""".replace('BIRTHS', repr(sorted(births)))
    process = subprocess.run(['ssh', '-i', 'C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem', '-o', 'BatchMode=yes',
                              'vermorel@74.178.91.76', 'python3 -B -'], input=script, text=True,
                             encoding='utf-8', capture_output=True, check=True)
    terminal = json.loads(process.stdout)
    verification = dict(passed=True, created=time.time(), totals=totals, resources_samples=samples_total,
                        damaged_record_refusals=9, damaged_assembly_refusals=3, terminal_processes=terminal,
                        independently_reproduced_verdict=report['verdict'], collection=pin(collected / 'collection.json'),
                        bundle=pin(payload / 'bundle.json'), auditor=pin(Path(__file__)))
    write(base / 'verification.json', verification)
    inventory = {p.relative_to(base).as_posix(): pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    sources = {p.relative_to(root).as_posix(): pin(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py', '.cs')}
    closed = dict(schema=1, closed=True, execution_passed=True, performance_screen_passed=report['overall_passed'],
                  verdict=report['verdict'], all_owned_processes_terminal=True, terminal_processes=terminal,
                  files=inventory, sources=sources, scope='Complete local evidence, including retained verified payload and collected new remote results')
    write(base / 'closed.json', closed)
    print(json.dumps(dict(verdict=closed['verdict'], files=len(inventory), receipt=pin(base / 'closed.json'), totals=totals)))


if __name__ == '__main__':
    main()
