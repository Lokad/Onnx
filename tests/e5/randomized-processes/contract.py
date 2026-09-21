"""Reuse the qualified v1 worker contract under the new v2 campaign schedule."""
from pathlib import Path
from fractions import Fraction
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'process-uncertainty'))
from worker_audit import pin, read, write, worker, telemetry
from protocol import CORE, LIMITS, CRITERIA, specification as worker_specification
from design import PROTOCOL, COHORTS, CASES, POLICIES, assignment_schedule, fieller, student_critical, process_mean


def manifest(base):
    meta = read(base/'frozen.json')
    assert meta['protocol'] == PROTOCOL and meta['worker_protocol'] == 'e5-fresh-process-uncertainty-v1'
    assert meta['mode'] in ('smoke', 'full') and meta['limits'] == LIMITS and meta['criteria'] == CRITERIA
    assert meta['core_sha256'] == CORE
    for name, want in meta['files'].items():
        assert pin(base/name) == want, name
    assignments = read(base/'assignments-frozen.json')
    assert assignments['protocol'] == PROTOCOL and assignments['cohorts'] == COHORTS
    assert assignments['inference_started'] is False
    for phase in ('aa', 'compare'):
        jobs = assignment_schedule(assignments['draws'][phase], phase)
        assert jobs == assignments['schedules'][phase]
        if meta['mode'] == 'smoke':
            jobs = [j for j in jobs if j['cohort'] == 0 and
                    (j['case_index'], j['policy']) in ((0, 'default'), (4, 'memory'))]
        assert meta['schedules'][phase] == jobs
    for key in ('model', 'native', 'dotnet'):
        item = meta[key]
        assert pin(Path(item['path'])) == {k: item[k] for k in ('bytes', 'sha256')}, key
    for item in meta['runtime_files']:
        assert pin(Path(item['path'])) == {k: item[k] for k in ('bytes', 'sha256')}, item['path']
    return meta


def worker_value(base, phase, meta, job):
    return worker(base/('result-'+phase)/job['name']/'output', base/'inputs', meta['model'],
                  meta['files']['bin/ProcessUncertainty.dll'], meta['files']['bin/Microsoft.ML.OnnxRuntime.dll'],
                  meta['native'], job, phase, meta['mode'] == 'smoke')


def timing(values, jobs, phase):
    assert len(values) == len(jobs) == 1800
    alpha, contrasts = .05, ([('C', 'A')] if phase == 'aa' else [('C', 'A'), ('C', 'N'), ('A', 'N')])
    critical = student_critical(1-alpha/(20*len(contrasts)), COHORTS-1)
    groups = {}
    for value, job in zip(values, jobs, strict=True):
        assert value['specification'] == worker_specification(job, phase)
        key = (job['case_index'], job['policy'], job['cohort'], job['role'])
        assert key not in groups
        groups[key] = {b: process_mean(value['measured'], b, value['frequency']) for b in ('execute', 'request')}
    results, passes = [], True
    for index, case in enumerate(CASES):
        for policy in POLICIES:
            for boundary in ('execute', 'request'):
                estimates = {}
                for numerator, denominator in contrasts:
                    y = [groups[(index, policy, c, numerator)][boundary] for c in range(COHORTS)]
                    x = [groups[(index, policy, c, denominator)][boundary] for c in range(COHORTS)]
                    result = fieller(y, x, critical)
                    fitted_ratio = sum(y)/sum(x)
                    differences = [a-fitted_ratio*b for a, b in zip(y, x, strict=True)]
                    average = sum(differences)/COHORTS
                    squares = [(d-average)**2 for d in differences]
                    result['largest_observed_variance_share'] = float(max(squares)/sum(squares)) if sum(squares) else 0.
                    result['process_means'] = dict(numerator=[float(v) for v in y], denominator=[float(v) for v in x])
                    estimates[numerator+'/'+denominator] = result
                primary = estimates['C/A']; bounds = primary['interval']
                if phase == 'aa':
                    passed = primary['bounded'] and .99 <= bounds[0] <= 1 <= bounds[1] <= 1.01
                else:
                    passed = primary['bounded'] and 0 <= bounds[0] <= bounds[1] <= CRITERIA['candidate_upper'][index]
                passes &= passed
                results.append(dict(case=case, policy=policy, boundary=boundary, contrasts=estimates,
                                    statistical_screen=passed, primary=index < 4))
    return dict(statistical_screen=passes, results=results, critical=critical,
                family_alpha=alpha, contrast_count=20*len(contrasts),
                scope='Approximate fixed-campaign randomization intervals; no-interference and regularity assumptions apply.',
                ready_for_promotion=False)


def audit(base, phase):
    meta = manifest(base); jobs = meta['schedules'][phase]; folder = base/('result-'+phase)
    state = read(folder/'identity.json')
    assert state['phase'] == phase and state['frozen'] == pin(base/'frozen.json')
    samples = {j['name']: [json.loads(line) for line in (folder/j['name']/'samples.jsonl').read_text().splitlines()] for j in jobs}
    resources = telemetry(state, samples, jobs)
    if meta['mode'] == 'full':
        sys.path.insert(0, str(base/'eng'))
        import campaign_processes as accounting
        for run, resource in zip(state['runs'], resources['workers'], strict=True):
            directory = folder/run['job']['name']
            observed = accounting.foreign_fraction(read(directory/'pre.json'), read(directory/'post.json'), state['supervisor']['pid'])
            assert observed == run['accounting'] and observed['foreign_cpu_fraction'] <= .02
            def cpu(path):
                return [int(v) for v in path.read_text().splitlines()[0].split()[1:]]
            delta = [b-a for a, b in zip(cpu(directory/'cpu-before.txt'), cpu(directory/'cpu-after.txt'), strict=True)]
            assert len(delta) >= 8 and all(v >= 0 for v in delta) and sum(delta[:8]) > 0
            steal = delta[7]/sum(delta[:8]); assert steal <= .005
            resource.update(foreign=observed, steal=steal)
            for sample in samples[run['job']['name']]:
                assert sample['disk_free'] >= 512*1024**2
                for member in sample['members']:
                    assert member['threads'] and all(t['affinity'] == [2] for t in member['threads'])
    values = [worker_value(base, phase, meta, job) for job in jobs]
    for index in set(j['case_index'] for j in jobs):
        for native in (False, True):
            group = [v for v in values if v['specification']['case_index'] == index and (v['specification']['role'] == 'N') == native]
            assert len({v['output_sha256'] for v in group}) == 1
    return dict(passed=True, phase=phase, mode=meta['mode'], frozen=pin(base/'frozen.json'),
                identity=pin(folder/'identity.json'), resources=resources,
                measured_calls=sum(len(v['measured']) for v in values), conditioning_calls=sum(len(v['conditioning']) for v in values),
                maximum_native_error=max(max(v['before_error'], v['after_error']) for v in values),
                timing=timing(values, jobs, phase) if meta['mode'] == 'full' else None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--phase', choices=['aa', 'compare'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = audit(args.payload.resolve(), args.phase)
    write(args.output, result)
    print(json.dumps(dict(passed=True, mode=result['mode'], phase=args.phase, measured_calls=result['measured_calls'],
                          statistical_screen=result['timing']['statistical_screen'] if result['timing'] else None)))
