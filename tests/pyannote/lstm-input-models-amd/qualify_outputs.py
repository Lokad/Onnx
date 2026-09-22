"""Audit every AMD graph tensor and public result before allowing timing."""
import collections
import math
from pathlib import Path
import numpy as np
from candidate_protocol import below, pin, read


def scaled(actual, expected):
    assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))/np.maximum(1., np.abs(expected.astype(np.float64)))
    return dict(values=int(actual.size), failed_values=int(np.count_nonzero(delta > 1e-4)),
                maximum=float(delta.max(initial=0)), bit_identical=actual.tobytes() == expected.tobytes())


def pyannote(base, folder, role, baseline=None):
    manifest_path = base/'manifests'/(role+'-pyannote.json')
    manifest = read(manifest_path); result = read(folder/'result.json')
    assert result['passed'] is True and result['inputs_and_held_outputs_unchanged'] is True
    assert result['runtime'] == '10.0.8'
    assert result['manifest_sha256'] == pin(manifest_path)['sha256']
    for key in ('core_sha256', 'data_sha256'):
        assert result[key] == manifest[key]
    crops = [c['name'] for c in manifest['cases'] if c['samples'] == 160000]
    assert len(crops) == 3
    assert [(r['name'], r['model'], r['pass']) for r in result['rows']] == [
        (c, m, p) for c in crops for m in ('segmentation', 'embedding') for p in range(3)]
    references = {(r['name'], r['model']): r for r in read(base/'graph-reference.json')}
    baseline_result = read(baseline/'result.json') if baseline is not None else None
    files = set(); groups = collections.defaultdict(list); comparisons = []; profiles = []
    for index, row in enumerate(result['rows']):
        values = {}
        for kind in ('input', 'output'):
            item = row[kind]; path = below(folder, item['file']); values[kind] = np.fromfile(path, dtype='<f4').reshape(item['shape'])
            assert pin(path) == dict(bytes=item['values']*4, sha256=item['sha256'])
            assert values[kind].size == item['values'] and np.isfinite(values[kind]).all()
            files.add(item['file'])
        groups[(row['name'], row['model'])].append(row)
        ref = references[(row['name'], row['model'])]; path = below(base, ref['path'])
        assert pin(path) == {k: ref[k] for k in ('bytes', 'sha256')}
        expected = np.load(path, allow_pickle=False); assert expected.dtype == np.float32
        comparisons.append(dict(name=row['name'], model=row['model'], pass_index=row['pass'],
                                reference='native', **scaled(values['output'], expected)))
        if baseline_result is not None:
            other = baseline_result['rows'][index]
            assert (row['name'], row['model'], row['pass']) == (other['name'], other['model'], other['pass'])
            assert row['input'] == other['input'], 'Changed graph inputs across roles'
            expected = np.fromfile(below(baseline, other['output']['file']), dtype='<f4').reshape(other['output']['shape'])
            comparisons.append(dict(name=row['name'], model=row['model'], pass_index=row['pass'],
                                    reference='production', **scaled(values['output'], expected)))
        assert math.isfinite(row['seconds']) and row['seconds'] == (row['end_ticks']-row['start_ticks'])/row['frequency'] > 0
        assert len(row['nodes']) == ((108 if row['model'] == 'segmentation' else 75) if row['pass'] == 2 else 0)
        previous = row['start_ticks']; operators = collections.defaultdict(float)
        for node in row['nodes']:
            assert previous <= node['start_ticks'] <= node['end_ticks'] <= row['end_ticks']
            assert node['seconds'] == (node['end_ticks']-node['start_ticks'])/row['frequency']
            previous = node['end_ticks']; operators[node['op']] += node['seconds']
        if row['pass'] == 2:
            profiles.append(dict(name=row['name'], model=row['model'], graph_seconds=row['seconds'], operators=dict(operators)))
    for group in groups.values():
        assert len({r['input']['sha256'] for r in group}) == len({r['output']['sha256'] for r in group}) == 1
    assert len(files) == 24 and {p.name for p in folder.iterdir()} == files | {'result.json'}
    assert [(a['name'], a['pass'], a['phase']) for a in result['applications']] == [
        (c['name'], p, 'warmup' if p == 0 else 'measured') for p in range(4) for c in manifest['cases']]
    first = {}
    for row in result['applications']:
        actual = row['result']; expected = next(c['expected'] for c in manifest['cases'] if c['name'] == row['name'])
        assert actual['Status'] == 0 and expected['status'] == 'Completed'
        assert actual['Windows'] == expected['windows'] and actual['AudioDuration'] == expected['audio_seconds']
        for left, right in [('Intervals', 'intervals'), ('ExclusiveIntervals', 'exclusive_intervals')]:
            assert len(actual[left]) == len(expected[right])
            for a, b in zip(actual[left], expected[right], strict=True):
                assert a['Speaker'] == b[2] and abs(a['Start']-b[0]) <= 1e-12 and abs(a['End']-b[1]) <= 1e-12
        maximum = 0.
        for a, b in zip(actual['Speakers'], expected['speakers'], strict=True):
            assert a['Speaker'] == b['speaker'] and a['HasEmbedding'] == b['has_embedding']
            left = np.array(a['Centroid']); right = np.array(b['centroid']); assert left.shape == right.shape == (256,)
            check = scaled(left, right); assert check['failed_values'] == 0; maximum = max(maximum, check['maximum'])
        assert maximum == row['maximum_centroid_error'] and math.isfinite(row['seconds']) and row['seconds'] > 0
        if row['name'] in first:
            assert actual == first[row['name']], 'Changed public repeat'
        else:
            first[row['name']] = actual
    return dict(passed=all(c['failed_values'] == 0 for c in comparisons), arrays=18,
                values=sum(r['output']['values'] for r in result['rows']), public_calls=16,
                comparisons=comparisons, profiles=profiles, result=pin(folder/'result.json'))


def parakeet(base, result_path, role):
    # This exact, previously used independent audit is included in the execution bundle.
    from parakeet_audit import audit
    result = read(result_path); manifest = read(base/'manifests'/(role+'-parakeet.json'))
    assert result['runtime'] == '.NET 10.0.8' and not result['settings']
    for key in ('core_sha256', 'data_sha256'):
        assert result[key] == manifest[key]
    assert result['runner_sha256'] == pin(base/'runtimes'/role/'TranscribeReplay.dll')['sha256']
    return audit(base/'parakeet-reference/manifest.json', result_path)
