"""Check the selected compiled scope, full LSTM census and loaded products."""
import xml.etree.ElementTree as ET
from protocol import read, pin


def census(path, lstm_only):
    root = ET.parse(path).getroot(); ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    rows = root.findall('.//t:UnitTestResult', ns)
    if lstm_only: rows = [r for r in rows if 'lstm' in r.attrib['testName'].lower()]
    result = {r.attrib['testName']: r.attrib['outcome'] for r in rows}
    assert rows and len(result) == len(rows)
    return result


def compiled(value, baseline, candidate):
    assert value['inventory_complete'] and len(value['observations']) == 2
    details = []
    for row in value['observations']:
        core = row['assembly'] == 'Lokad.Onnx.dll'
        assert core or row['assembly'] == 'Lokad.Onnx.Data.dll'
        assert row['before_sha256'] == baseline[row['assembly']]['sha256']
        assert row['after_sha256'] == candidate[row['assembly']]['sha256']
        assert row['methods'] == (3284 if core else 697)
        assert row['public_surface_equal'] and row['public_surface'] == row['public_surface_after']
        assert row['assembly_attributes_before'] == row['assembly_attributes_after'] and not row['removed']
        assert {k: row['method_flags_after'][k] for k in row['method_flags_before']} == row['method_flags_before']
        if core:
            assert len(row['differences']) == 2
            assert {tuple(n.split('::')[:2]) for n in row['differences']} == {
                ('Lokad.Onnx.GraphLstmPacking', 'Prepare'), ('Lokad.Onnx.CPUExecutionProvider', 'Lstm')}
            assert len(row['added']) == 2
            assert {tuple(n.split('::')[:2]) for n in row['added']} == {
                ('Lokad.Onnx.GraphLstmPacking', 'get_ColumnsPerBlock'),
                ('Lokad.Onnx.CPUExecutionProvider', 'LstmProjectPreparedOrdered')}
            assert row['unchanged_methods'] == 3282
        else:
            assert not row['differences'] and not row['added'] and row['unchanged_methods'] == 697
        details.append(dict(assembly=row['assembly'], unchanged=row['unchanged_methods'], changed=row['differences'], added=row['added']))
    return dict(passed=True, methods=details, public_surface_equal=True, flags_equal=True, attributes_equal=True)


def contracts(folder, spec, built, row):
    mode = row['name'].removeprefix('contracts-')
    loaded, projections = read(folder/'loaded.json'), read(folder/'projections.json')
    assert loaded['passed'] and loaded['mode'] == mode
    assert str(loaded['pid']) in row['members']
    assert loaded['core_sha256'] == built['products']['candidate']['sha256']
    assert loaded['consumer_sha256'] == built['consumer']['sha256']
    assert loaded['avx512'] == (mode == 'normal') and loaded['hardware'] == (mode != 'scalar')
    assert loaded['runtime'] == '10.0.8' and loaded['affinity'] == 4 and loaded['block'] == 4 * loaded['vector_count']
    assert projections['passed'] and projections['calls'] == 380 and projections['projections'] == 760
    assert projections['values'] == 1945600 and len(projections['hashes']) == 760
    actual = census(folder/'contracts.trx', False)
    expected = spec['expected_census']
    # All LSTM cases in the parent pass on normal and AVX512-disabled execution.
    assert set(actual) == set(expected) | set(spec['added_tests'])
    assert all(outcome == 'Passed' for outcome in actual.values())
    assert all(outcome == 'Passed' for outcome in expected.values())
    return dict(passed=True, mode=mode, passed_tests=len(actual), skipped_tests=0,
        census_exact=True, loaded=loaded, projections=projections, trx=pin(folder/'contracts.trx'))
