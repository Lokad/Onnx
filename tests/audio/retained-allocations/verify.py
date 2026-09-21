"""Independently compare every derived counter observation with its bound raw row."""
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/audio-retained-allocations-20260921'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def require(condition, description):
    if not condition:
        raise ValueError(description)


def main():
    closure = read(BASE / 'closed.json')
    require(pin(BASE / 'closed.json')['sha256'] == '54c5e08b2976a23dcfdbe02798fde5a147f870fc6f0db59d482a941029ad8ded', 'Analysis closure')
    for name, expected in closure['outputs'].items():
        require(pin(BASE / name) == expected, 'Analysis output identity')
    for name, expected in closure['tools'].items():
        require(pin(Path(name)) == expected, 'Executed tool identity')
    document = read(BASE / 'observations.json')
    summary = read(BASE / 'summary.json')
    require(len(document['observations']) == 3047, 'All observations retained')
    seen = set()
    grouped = defaultdict(list)
    allocations = collections = 0
    for meta in document['metadata']:
        result_path = ROOT / meta['result']
        require(pin(result_path) == meta['result_identity'], 'Raw result identity')
        result = read(result_path)
        if meta['scope'] == 'graph':
            raw = []
            for name in result['call_files']:
                path = result_path.parent / name
                require(pin(path) == closure['files'][str(path.relative_to(ROOT))], 'Raw graph identity')
                raw.append(read(path))
        else:
            raw = result.get('records', result.get('applications'))
        derived = [r for r in document['observations'] if all(r[k] == meta[k] for k in ['source', 'worker', 'scope'])]
        require(len(derived) == len(raw), 'Per-process coverage')
        for index, (item, original) in enumerate(zip(derived, raw, strict=True)):
            identity = (item['source'], item['worker'], item['scope'], index)
            require(identity not in seen and item['ordinal'] == index, 'No duplicate or omitted ordinal')
            seen.add(identity)
            require(item['name'] == original['name'], 'Input name')
            require(item['phase'] == original.get('phase', 'single') and item['pass_index'] == original.get('pass'), 'Phase and pass')
            for key in ['start_ticks', 'end_ticks', 'frequency', 'allocated_bytes', 'gc_before', 'gc_after', 'graph', 'step', 'input_sha256']:
                require(item.get(key) == original.get(key), 'Raw field: ' + key)
            require(item['seconds'] == float(Fraction(original['end_ticks'] - original['start_ticks'], original['frequency'])), 'Clock interval')
            if 'allocated_bytes' in original:
                allocations += 1
            if 'gc_before' in original:
                collections += 1
                require(item['gc_delta'] == [original['gc_after'][g] - original['gc_before'][g] for g in range(3)], 'Count differences')
            key = tuple(item[k] for k in ['source', 'role', 'scope', 'phase']) + (item.get('graph', item['name']),)
            grouped[key].append(original)
    require(len(seen) == 3047 and allocations == 2855 and collections == 2835, 'All counter coverage')
    require(len(grouped) == len(summary['groups']), 'Group coverage')
    for aggregate in summary['groups']:
        key = tuple(aggregate[k] for k in ['source', 'role', 'scope', 'phase', 'name'])
        rows = grouped[key]
        require(aggregate['calls'] == len(rows), 'Group calls')
        if 'allocated_bytes_sum' in aggregate:
            require(aggregate['allocated_bytes_sum'] == sum(r['allocated_bytes'] for r in rows), 'Allocation sum')
            require(aggregate['allocated_bytes_min'] == min(r['allocated_bytes'] for r in rows), 'Allocation minimum')
            require(aggregate['allocated_bytes_max'] == max(r['allocated_bytes'] for r in rows), 'Allocation maximum')
        if 'gc_delta_sum' in aggregate:
            require(aggregate['gc_delta_sum'] == [sum(r['gc_after'][g] - r['gc_before'][g] for r in rows) for g in range(3)], 'Collection sums')

    # This source archive produced the exact candidate used by both pyannote runs.
    archive = ROOT / 'artifacts/pyannote-lstm-output-lanes-20260921'
    receipt = archive / 'qualification-closed.json'
    require(pin(receipt)['sha256'] == 'cf950ec5cedf702c1af38decc377cd516a5d8b652f77853d03d2c81db0b5bf53', 'Candidate qualification identity')
    qualified = read(receipt)
    source_files = {}
    for name in ['src/Lokad.Onnx.Data/Community1Diarizer.cs', 'src/Lokad.Onnx.Data/WeSpeakerEmbedder.cs',
        'src/Lokad.Onnx/GraphExecution.cs', 'src/Lokad.Onnx/ReleasedBufferCache.cs',
        'src/Lokad.Onnx/TensorBufferPool.cs', 'src/Lokad.Onnx/ComputationalGraph.cs', 'src/Lokad.Onnx/AblationSwitches.cs']:
        path = archive / 'candidate-source' / name
        identity = pin(path)
        require(identity == qualified['files'][str(path.relative_to(ROOT))], 'Qualified source: ' + name)
        require(path.read_text(encoding='utf8') == (ROOT / name).read_text(encoding='utf8'), 'Working source differs beyond newline encoding: ' + name)
        source_files[name] = identity
    require(summary['original_performance_verdict'] == dict(attribution_valid=False, selected_for_amd=None), 'Original timing verdict')
    print(json.dumps(dict(passed=True, raw_observations_checked=len(seen), allocation_rows=allocations,
        collection_rows=collections, qualified_source_files=source_files,
        scope='Independently verified all derived rows and grouped counter totals; original full file maps were checked by analyze.py.'), indent=2))


if __name__ == '__main__':
    main()
