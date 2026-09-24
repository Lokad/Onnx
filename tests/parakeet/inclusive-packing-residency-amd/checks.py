"""Check every model mapping, ownership result and aggregate budget independently."""
import re


def residency(value, name, spec, built, run):
    role, graph, mode = name.split('-')
    assert value['passed'] and value['pid'] == run['child']['pid']
    assert value['model'] == spec['models'][graph]['path']
    assert value['budget'] == spec['budgets'][graph] == (256 if graph == 'encoder' else 64) * 1024**2
    assert value['runtime'] == '.NET 10.0.8' and value['affinity'] == 4 and value['processor_count'] == 1
    assert value['avx512'] == (mode == '512') and value['avx2'] and value['fma']
    assert value['core_sha256'] == spec['identities'][role]['Lokad.Onnx.dll']['sha256']
    assert value['data_sha256'] == spec['identities'][role]['Lokad.Onnx.Data.dll']['sha256']
    assert value['runner_sha256'] == built['consumer']['sha256']
    assert value['invalidation_rebuild_passed'] and value['weights'] == value['rebuilt_weights']
    rows = value['weights']; names = [r['name'] for r in rows]
    assert names == sorted(set(names))
    assert len(set(r['packed_name'] for r in rows)) == len(rows)
    for row in rows:
        assert row['packed_name'] == 'packed:' + row['name']
        assert row['source_initializer_keys'] and row['source_initializer_keys'] == sorted(set(row['source_initializer_keys']))
        assert len(row['shape']) == 2 and all(type(n) is int and n > 0 for n in row['shape'])
        reduction, width = row['shape']
        assert reduction <= (4095 if role == 'selected' else 4096)
        assert row['bytes'] == reduction * width * 4 <= 512 * 1024**2
        assert all(re.fullmatch('[a-f0-9]{64}', row[k]) for k in ['source_sha256', 'packed_sha256'])
    assert sum(r['bytes'] for r in rows) == value['retained_bytes'] <= value['budget']
    return dict(passed=True, role=role, graph=graph, avx512=mode == '512', weights=len(rows),
                retained_bytes=value['retained_bytes'], budget=value['budget'],
                inclusive_boundary_weights=sum(r['shape'][0] == 4096 for r in rows),
                source_and_clone_ownership=True, invalidation_rebuild=True)


def compare(values):
    for role in ['selected', 'candidate']:
        for graph in ['encoder', 'decoder']:
            a, b = [values[f'{role}-{graph}-{mode}'] for mode in ['512', '256']]
            assert a['weights'] == b['weights'] and a['retained_bytes'] == b['retained_bytes']
    differences = {}
    for graph in ['encoder', 'decoder']:
        old = {r['name']: r for r in values[f'selected-{graph}-512']['weights']}
        new = {r['name']: r for r in values[f'candidate-{graph}-512']['weights']}
        for name in old.keys() & new.keys(): assert old[name] == new[name], name
        differences[graph] = dict(selected=len(old), candidate=len(new),
                                 retained=sorted(old.keys() & new.keys()),
                                 displaced=sorted(old.keys() - new.keys()),
                                 newly_retained=sorted(new.keys() - old.keys()))
    return dict(passed=True, both_modes_exact=True, shared_source_and_packed_bytes_exact=True, graphs=differences)
