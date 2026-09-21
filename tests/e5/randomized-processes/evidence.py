"""Local checks of collected arrays, resources and complete phase statistics."""
from fractions import Fraction
from pathlib import Path
import json
import statistics
import sys

from contract import pin, read, write, worker, telemetry, timing, CORE, LIMITS, CRITERIA
from design import PROTOCOL, CASES, POLICIES, ROLES, assignment_schedule, process_mean


def verify_inventory(base, phase):
    receipt = read(base/(phase+'-collection.json'))
    meta = read(base/'frozen.json')
    assert receipt['terminal'] is True and receipt['phase'] == phase
    assert receipt['frozen'] == pin(base/'frozen.json')
    for name, identity in receipt['files'].items():
        assert pin(base/name) == identity, name
    assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()} == set(receipt['files']) | {phase+'-collection.json'}
    assert meta['protocol'] == PROTOCOL and meta['mode'] == 'full' and meta['core_sha256'] == CORE
    assert meta['limits'] == LIMITS and meta['criteria'] == CRITERIA
    assert meta['diagnostic_limits'] == dict(maximum_observed_variance_share=.2)
    for name, wanted in meta['files'].items():
        assert receipt['files'][name] == wanted
    expected = {item['path']: {k: item[k] for k in ('bytes', 'sha256')}
                for item in [meta['model'], meta['native'], meta['dotnet']]+meta['runtime_files']}
    expected.update(meta['python_files']); expected[meta['python_executable']] = meta['interpreter']
    assert receipt['external'] == expected
    assignments = read(base/'assignments-frozen.json')
    assert pin(base/'assignments-frozen.json') == dict(bytes=889069, sha256='2ac2bd42acfc811b39f2df099c7e93f31789bea18236e0e507d74e60ba5b16f5')
    for name in ('aa', 'compare'):
        assert meta['schedules'][name] == assignments['schedules'][name] == assignment_schedule(assignments['draws'][name], name)
    return meta, receipt


def summarize(values, jobs):
    """Retain all-sample statistics and ordering diagnostics outside the CI layer."""
    rows = []
    for index, case in enumerate(CASES):
        for policy in POLICIES:
            for role in ROLES:
                group = [(j, v) for j, v in zip(jobs, values, strict=True) if (j['case_index'], j['policy'], j['role']) == (index, policy, role)]
                assert len(group) == 60
                boundaries = {}
                for boundary in ('execute', 'request'):
                    samples = [Fraction(r[boundary], v['frequency']) for j, v in group for r in v['measured']]
                    means = [process_mean(v['measured'], boundary, v['frequency']) for j, v in group]
                    ordered = sorted(samples)
                    positions = {str(p): [m for (j, v), m in zip(group, means, strict=True) if j['position'] == p] for p in range(3)}
                    boundaries[boundary] = dict(mean_ms=float(sum(means)/60)*1000, median_ms=float(statistics.median(ordered))*1000,
                        p95_ms=float(ordered[(len(ordered)*95+99)//100-1])*1000, maximum_ms=float(ordered[-1])*1000,
                        position_mean_ms={p: float(sum(s)/len(s))*1000 if s else None for p, s in positions.items()},
                        position_counts={p: len(s) for p, s in positions.items()},
                        first_half_mean_ms=float(sum(means[:30])/30)*1000, second_half_mean_ms=float(sum(means[30:])/30)*1000)
                measured = [r for j, v in group for r in v['measured']]
                rows.append(dict(case=case, policy=policy, role=role, workers=60, measured_calls=len(measured),
                    conditioning_calls=sum(len(v['conditioning']) for j, v in group), boundaries=boundaries,
                    load_median_ms=statistics.median(v['load_ticks']/v['frequency']*1000 for j, v in group),
                    first_execute_median_ms=statistics.median(v['first']['execute']/v['frequency']*1000 for j, v in group),
                    measured_gc=[sum(r['g'+str(g)] for r in measured) for g in range(3)],
                    measured_allocated_bytes=sum(r['bytes'] for r in measured)))
    return rows


def audit_collected(base, phase):
    meta, receipt = verify_inventory(base, phase)
    assert receipt['code'] == 0 and receipt['remote_audit_code'] == 0, 'Failed phases are retained, not scored'
    jobs = meta['schedules'][phase]; folder = base/('result-'+phase)
    state = read(folder/'identity.json')
    samples = {j['name']: [json.loads(line) for line in (folder/j['name']/'samples.jsonl').read_text().splitlines()] for j in jobs}
    resources = telemetry(state, samples, jobs)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'eng'))
    import campaign_processes as accounting
    assert pin(Path(accounting.__file__)) == meta['files']['eng/campaign_processes.py']
    for run, resource in zip(state['runs'], resources['workers'], strict=True):
        directory = folder/run['job']['name']
        foreign = accounting.foreign_fraction(read(directory/'pre.json'), read(directory/'post.json'), state['supervisor']['pid'])
        assert foreign == run['accounting'] and foreign['foreign_cpu_fraction'] <= .02
        before = [int(v) for v in (directory/'cpu-before.txt').read_text().splitlines()[0].split()[1:]]
        after = [int(v) for v in (directory/'cpu-after.txt').read_text().splitlines()[0].split()[1:]]
        delta = [b-a for a, b in zip(before, after, strict=True)]
        assert len(delta) >= 8 and all(v >= 0 for v in delta) and sum(delta[:8]) > 0
        steal = delta[7]/sum(delta[:8]); assert steal <= .005
        resource.update(foreign=foreign, steal=steal)
        for sample in samples[run['job']['name']]:
            assert sample['disk_free'] >= 512*1024**2
            for member in sample['members']:
                assert member['threads'] and all(t['affinity'] == [2] for t in member['threads'])
    values = [worker(folder/j['name']/'output', base/'inputs', meta['model'], meta['files']['bin/ProcessUncertainty.dll'],
                     meta['files']['bin/Microsoft.ML.OnnxRuntime.dll'], meta['native'], j, phase) for j in jobs]
    for index in range(5):
        for native in (False, True):
            group = [v for v in values if v['specification']['case_index'] == index and (v['specification']['role'] == 'N') == native]
            assert len({v['output_sha256'] for v in group}) == 1
    result = dict(passed=True, phase=phase, mode='full', frozen=pin(base/'frozen.json'), identity=pin(folder/'identity.json'),
                  resources=resources, measured_calls=sum(len(v['measured']) for v in values),
                  conditioning_calls=sum(len(v['conditioning']) for v in values),
                  maximum_native_error=max(max(v['before_error'], v['after_error']) for v in values), timing=timing(values, jobs, phase))
    assert result == read(base/(phase+'-remote-audit.json')), 'Local and remote raw audits disagree'
    result['summaries'] = summarize(values, jobs)
    result['diagnostic_screen'] = all(c['largest_observed_variance_share'] <= .2 for row in result['timing']['results'] for c in row['contrasts'].values())
    result['collection'] = pin(base/(phase+'-collection.json'))
    return result
