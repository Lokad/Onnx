"""Retain the terminal first review rejection without modifying its inputs."""
import json
from prepare import BASE, pin, read, save, verify, terminal


def main():
    assert not (BASE/'failure-closed.json').exists()
    value = read(BASE/'inputs.json'); verify(value['files'])
    state = read(BASE/'preparation.json'); assert state['complete'] and state['code'] == 1
    assert 'AssertionError' in state['error'] and 'review()' in state['error']
    assert [r['name'] for r in state['runs']] == [n+'-'+phase for n in ['cli', 'backend', 'tensors', 'bridge'] for phase in ['restore', 'build']] + ['inventory']
    identities = [state['supervisor']]; resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < 900
        assert row['preflight']['available'] >= 8*1024**3
        samples = [json.loads(s) for s in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            assert sample['seconds'] < 900 and sample['rss'] < 8*1024**3 and sample['available'] >= 1024**3
            assert sample['disk'] >= 20*1024**3 and sample['output_bytes'] <= 1024**3
            assert sample['rss'] == sum(m['rss'] for m in sample['members'])
            for m in sample['members']: assert m['affinity'] == [2] and row['members'][str(m['pid'])] == m['birth']
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: terminal(identity)
    inventory = read(BASE/'instructions.json'); assert inventory['inventory_complete']
    core, data = inventory['observations']; assert core['public_surface_equal'] and data['public_surface_equal']
    assert len(core['removed']) == 190 and all('<' in k.split('::')[0] or k.split('::')[1].startswith('<') for k in core['removed'])
    assert data['unchanged_methods'] == data['methods'] == 697
    save(BASE/'failure-analysis.json', dict(passed=False, retained_failure=True, normal_projects_built=True,
        public_surface_equal=True, data_methods_unchanged=697, unnormalized_removed_core_methods=190,
        focused_tests_executed=False, compiler_rename_proof_pending=True, resources=resources, error=state['error']))
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'failure-closed.json', dict(passed=False, retained_failure=True, files=files, local_inputs=value['files'], identities=identities,
        core=pin(BASE/'runtime/Lokad.Onnx.dll'), data=pin(BASE/'runtime/Lokad.Onnx.Data.dll')))
    print(json.dumps(dict(closed=pin(BASE/'failure-closed.json'), core=pin(BASE/'runtime/Lokad.Onnx.dll'),
        data=pin(BASE/'runtime/Lokad.Onnx.Data.dll'), samples=sum(r['samples'] for r in resources))))


if __name__ == '__main__': main()
