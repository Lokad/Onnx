"""Read and bind retained audio counters; no model execution or timing retry."""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
DEFAULT = ROOT / 'artifacts/audio-retained-allocations-20260921'
SOURCES = {
    'pyannote-optimized-ort-20260921': 'c2e1d5da4746a6fb4cd4fe582663d922e074cb6bb4e03234757274cdfcf11a26',
    'pyannote-optimized-meetings-20260921': 'f9978dd22cec3091e640030e1db8b8e70aeb3c111c43d704c11f679ca5a89c1c',
    'parakeet-arithmetic-comparison-20260921': 'b7ed78b2c582fba0ef43c411540f501f3c997eeb73b77e1f8dc9a8574ea0b27b',
    'parakeet-performance-profile-v2-20260921': '6d8ce878f99acf291cb20348bba948bd1de7f128298c774cf94adf84e855700d',
}
LIMITATIONS = [
    'Post hoc descriptive analysis of retained counters, with no new model execution.',
    'Allocation values are approximate process-wide managed bytes, not RSS, live heap size or native allocations.',
    'Each GC generation count includes collections of older generations; do not add the three columns as distinct events.',
    'Count intervals bracket the timed call but are sampled separately; they are not pause durations or allocation stack traces.',
    'Graph counters include small instrumentation allocations and the graph trace retains outputs; do not subtract them from separate public runs.',
    'Missing native allocation counters are unknown, not zero.',
    'Original application, numerical, resource and performance verdicts remain unchanged; causality is unassigned.',
]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def save(path, value):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    check(not args.output.exists(), 'Refusing to overwrite an existing output directory')
    hashes = {}
    bound = {}
    sources = {}
    def verify(path, wanted):
        path = path.resolve()
        if path not in hashes:
            hashes[path] = pin(path)
        check(hashes[path] == wanted, 'File identity changed: ' + str(path))
        bound[path] = wanted
    for name, sha in SOURCES.items():
        base = ROOT / 'artifacts' / name
        receipt = base / 'closed.json'
        identity = pin(receipt)
        check(identity['sha256'] == sha, 'Closure changed: ' + name)
        verify(receipt, identity)
        closed = read(receipt)
        check(closed['passed'], 'Original evidence failed: ' + name)
        for filename, wanted in closed['files'].items():
            verify(ROOT / filename, wanted)
        for filename, wanted in closed.get('external_files', {}).items():
            verify(Path(filename), wanted)
        verify(base / 'analysis.json', closed['analysis'])
        sources[name] = dict(closure=identity, file_pins=len(closed['files']),
            external_pins=len(closed.get('external_files', {})), analysis=closed['analysis'])
        print('Verified', name, len(closed['files']), 'file pins', flush=True)

    observations = []
    metadata = []
    def append_rows(source, worker, role, scope, file, rows, counts=True, allocated=True, individual=False):
        check(file.resolve() in bound, 'Unbound result: ' + str(file))
        result = read(file)
        metadata.append(dict(source=source, worker=worker, role=role, scope=scope,
            result=file.relative_to(ROOT).as_posix(), result_identity=pin(file),
            runtime={key: result[key] for key in ['core_sha256', 'data_sha256', 'runner_sha256',
                'runtime', 'onnxruntime', 'engine', 'affinity', 'flags'] if key in result}))
        previous_gc = None
        previous_end = None
        for index, row in enumerate(rows):
            if individual:
                original = file.parent / f'{index:03}.json'
                check(original.resolve() in bound and read(original) == row, 'Individual row mismatch')
            start, end, frequency = (row[key] for key in ['start_ticks', 'end_ticks', 'frequency'])
            check(all(integer(v) for v in (start, end, frequency)) and end > start and frequency > 0, 'Invalid clock')
            check(previous_end is None or start >= previous_end, 'Overlapping requests')
            previous_end = end
            seconds = Fraction(end - start, frequency)
            check('seconds' not in row or row['seconds'] == float(seconds), 'Clock arithmetic changed')
            record = dict(source=source, worker=worker, role=role, scope=scope, ordinal=index,
                name=row['name'], phase=row.get('phase', 'single'), pass_index=row.get('pass'),
                start_ticks=start, end_ticks=end, frequency=frequency, seconds=float(seconds))
            for key in ('graph', 'step', 'input_sha256'):
                if key in row:
                    record[key] = row[key]
            if allocated:
                check(integer(row['allocated_bytes']), 'Invalid allocation count')
                record['allocated_bytes'] = row['allocated_bytes']
            else:
                check('allocated_bytes' not in row, 'Unexpected native allocation field')
            if counts:
                before, after = row['gc_before'], row['gc_after']
                check(len(before) == len(after) == 3 and all(integer(v) for v in before + after), 'Invalid GC counters')
                check(all(b >= a for a, b in zip(before, after)), 'Decreasing GC count')
                check(previous_gc is None or all(b >= a for a, b in zip(previous_gc, before)), 'Nonmonotone process counters')
                previous_gc = after
                record.update(gc_before=before, gc_after=after, gc_delta=[b - a for a, b in zip(before, after)])
            observations.append(record)

    name = 'pyannote-optimized-ort-20260921'
    base = ROOT / 'artifacts' / name
    for i, role in enumerate(['candidate', 'ort', 'ort', 'candidate']):
        file = base / 'process' / f'{i}-{role}' / 'output/result.json'
        result = read(file)
        check(len(result['records']) == 16 and result['held_outputs_unchanged'], 'Pyannote coverage')
        check([r['phase'] for r in result['records']] == ['warmup'] * 4 + ['measured'] * 12, 'Pyannote phases')
        append_rows(name, f'{i}-{role}', role, 'public', file, result['records'],
            counts=role != 'ort', allocated=role != 'ort', individual=True)

    name = 'pyannote-optimized-meetings-20260921'
    file = ROOT / 'artifacts' / name / 'output-run/result.json'
    result = read(file)
    check([r['name'] for r in result['records']] == ['ES2004a', 'IS1009a', 'ES2004a-recovery30'], 'Meeting coverage')
    check(result['held_outputs_unchanged'], 'Meeting ownership')
    for i, row in enumerate(result['records']):
        path = file.parent / f'{i:02}.json'
        check(path.resolve() in bound and read(path) == row, 'Meeting row mismatch')
    append_rows(name, 'run', 'candidate', 'public', file, result['records'])

    name = 'parakeet-arithmetic-comparison-20260921'
    base = ROOT / 'artifacts' / name
    original = read(base / 'analysis.json')
    check(original['attribution_valid'] is False and original['selected_for_amd'] is None, 'Original verdict changed')
    for i, role in enumerate(['production', 'candidate', 'ort', 'ort', 'candidate', 'production']):
        file = base / f'{i}-{role}' / 'output/result.json'
        result = read(file)
        check(len(result['records']) == 80 and result['held_outputs_unchanged'], 'Parakeet coverage')
        check([r['phase'] for r in result['records']] == ['warmup'] * 20 + ['measured'] * 60, 'Parakeet phases')
        append_rows(name, f'{i}-{role}', role, 'public', file, result['records'],
            counts=role != 'ort', allocated=role != 'ort', individual=True)

    name = 'parakeet-performance-profile-v2-20260921'
    base = ROOT / 'artifacts' / name
    file = base / 'trace-output/result.json'
    result = read(file)
    check(result['passed'] and result['inputs_and_held_outputs_unchanged'], 'Trace checks')
    check(result['call_files'] == [f'{i:04}.json' for i in range(2480)], 'Trace coverage')
    graph_rows = []
    for filename in result['call_files']:
        path = file.parent / filename
        check(path.resolve() in bound, 'Unbound graph row')
        graph_rows.append(read(path))
    check([r['phase'] for r in graph_rows] == ['unprofiled'] * 1240 + ['wall'] * 1240, 'Trace phases')
    append_rows(name, 'trace', 'production', 'graph', file, graph_rows)
    file = base / 'public-output/result.json'
    result = read(file)
    check(result['passed'] and result['inputs_and_held_outputs_unchanged'] and len(result['applications']) == 20, 'Profile public checks')
    append_rows(name, 'public', 'production', 'public-control', file, result['applications'], counts=False)

    def aggregate(rows):
        answer = dict(calls=len(rows), seconds_sum=sum(r['seconds'] for r in rows),
            seconds_mean=statistics.fmean(r['seconds'] for r in rows),
            seconds_min=min(r['seconds'] for r in rows), seconds_max=max(r['seconds'] for r in rows))
        if all('allocated_bytes' in r for r in rows):
            amounts = [r['allocated_bytes'] for r in rows]
            answer.update(allocated_bytes_sum=sum(amounts), allocated_bytes_mean=statistics.fmean(amounts),
                allocated_bytes_min=min(amounts), allocated_bytes_max=max(amounts))
        if all('gc_delta' in r for r in rows):
            answer.update(gc_delta_sum=[sum(r['gc_delta'][g] for r in rows) for g in range(3)],
                gc_delta_min=[min(r['gc_delta'][g] for r in rows) for g in range(3)],
                gc_delta_max=[max(r['gc_delta'][g] for r in rows) for g in range(3)],
                calls_without_gen2=sum(r['gc_delta'][2] == 0 for r in rows))
        return answer

    grouped = defaultdict(list)
    worker_groups = defaultdict(list)
    for row in observations:
        dimension = row.get('graph', row['name'])
        grouped[(row['source'], row['role'], row['scope'], row['phase'], dimension)].append(row)
        worker_groups[(row['source'], row['worker'], row['scope'], row['phase'])].append(row)
    groups = [dict(source=k[0], role=k[1], scope=k[2], phase=k[3], name=k[4], **aggregate(rows))
        for k, rows in grouped.items()]
    workers = [dict(source=k[0], worker=k[1], scope=k[2], phase=k[3], **aggregate(rows))
        for k, rows in worker_groups.items()]

    comparison = [r for r in observations if r['source'] == 'parakeet-arithmetic-comparison-20260921' and r['phase'] == 'measured']
    failed = []
    for role in ('production', 'candidate', 'ort'):
        rows = [r for r in comparison if r['role'] == role]
        names = list(dict.fromkeys(r['name'] for r in rows))
        for name in names:
            by_worker = defaultdict(list)
            clips = [r for r in rows if r['name'] == name]
            for r in clips:
                by_worker[r['worker']].append(Fraction(r['end_ticks'] - r['start_ticks'], r['frequency']))
            check(len(by_worker) == 2 and all(len(v) == 3 for v in by_worker.values()), 'Repeated clip coverage')
            means = {worker: sum(v) / len(v) for worker, v in by_worker.items()}
            ratio = max(means.values()) / min(means.values())
            check(abs(float(ratio) - original['controls'][role]['clip_max_min'][name]) < 1e-14, 'Original clip control changed')
            if ratio > Fraction(6, 5):
                failed.append(dict(role=role, name=name, ratio=float(ratio),
                    worker_means={k: float(v) for k, v in means.items()}, observations=clips))
    check([(v['role'], v['name']) for v in failed] == [
        ('production', '672-122797-0000'), ('production', '1221-135766-0002'),
        ('production', '1320-122612-0000'), ('candidate', '1188-133604-0001')], 'Failed control coverage')
    check(len(observations) == 3047, 'Observation coverage')
    summary = dict(passed=True, sources=sources, unique_verified_files=len(bound), observations=len(observations),
        with_allocation=sum('allocated_bytes' in r for r in observations),
        with_gc_counts=sum('gc_delta' in r for r in observations),
        groups=groups, workers=workers, failed_clip_controls=failed,
        original_performance_verdict=dict(attribution_valid=False, selected_for_amd=None), limitations=LIMITATIONS)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'observations.json', dict(metadata=metadata, observations=observations, limitations=LIMITATIONS))
    save(args.output / 'summary.json', summary)
    save(args.output / 'closed.json', dict(passed=True, sources=sources,
        files={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): value for p, value in bound.items()},
        tools={str(TOOLS / 'analyze.py'): pin(TOOLS / 'analyze.py')},
        outputs={name: pin(args.output / name) for name in ['observations.json', 'summary.json']},
        no_model_execution=True))
    print(json.dumps({key: summary[key] for key in ['passed', 'unique_verified_files', 'observations', 'with_allocation', 'with_gc_counts']}))
    print(json.dumps(dict(closed=pin(args.output / 'closed.json'))))


if __name__ == '__main__':
    main()
