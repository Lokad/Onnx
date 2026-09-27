"""Reuse exact node memberships while taking every clock from the new captures."""
import json
import math
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(1, str(TOOLS.parent/'pointwise-tail-profile-amd'))
from run import ROOT, load, pin, read

prior = load('reviewed_complete_mapping', TOOLS.parent/'owned-batch-isolation-profile-amd/diagnose.py')
descriptors_equal = prior.descriptors_equal
MAPPING = ROOT/'artifacts/parakeet-packed-final-row-gap-20260925'


def references():
    proof = read(MAPPING/'closed.json')
    assert pin(MAPPING/'closed.json')['sha256'] == '17f0e96862a89606b7e6f56ac5b3ba13fcd36507efa7388bf8c15fff5e0899a7'
    assert proof['passed'] and proof['analysis'] == pin(MAPPING/'analysis.json')
    assert proof['analysis']['sha256'] == '0faf7f224058af2600067d87e45e3b660f153ad67cc225623ba670f2fc0c7047'
    mapping = read(MAPPING/'analysis.json')
    values = {}
    for name, wanted in mapping['sources'].items():
        folder = ROOT/'artifacts'/name
        assert pin(folder/'closed.json') == wanted['closure']
        assert pin(folder/'analysis.json') == wanted['analysis']
        assert read(folder/'closed.json')['passed']
        values[name] = read(folder/'analysis.json')
    return (mapping, values['parakeet-packed-final-row-profile-closure-amd-20260925'],
            values['parakeet-ort-diagnosis-amd-20260924'])


def native_descriptors_equal(before, after):
    assert set(before) == set(after) == {'frontend','encoder','decoder'}
    def keyed(rows, key, ignored=()):
        result = {key(row): {k:v for k,v in row.items() if k not in ignored} for row in rows}
        assert len(result) == len(rows), 'Duplicate native descriptor'
        return result
    for graph in before:
        old, new = before[graph], after[graph]
        assert old['session_calls'] == new['session_calls'], graph
        assert old['nodes'] == new['nodes'], graph
        name = lambda row: row['name']
        assert keyed(old['node_clocks'], name, ('inclusive_us','exclusive_us')) == keyed(
            new['node_clocks'], name, ('inclusive_us','exclusive_us')), graph
        shape = lambda row: json.dumps({k:v for k,v in row.items() if k != 'calls'}, sort_keys=True)
        assert keyed(old['shapes'], shape) == keyed(new['shapes'], shape), graph


def partition(managed, native, mapping, old_managed, old_native):
    descriptors_equal(old_managed['phases']['wall']['node_rows'], managed['phases']['wall']['node_rows'])
    native_descriptors_equal(old_native['profiles'], native['profiles'])
    managed_rows = [r for r in managed['phases']['wall']['node_rows'] if r['graph'] == 'encoder']
    native_rows = native['profiles']['encoder']['node_clocks']
    nodes, ort = {r['name']:r for r in managed_rows}, {r['name']:r for r in native_rows}
    assert len(nodes) == len(managed_rows) == 2856 and len(ort) == len(native_rows) == 1993
    used, native_used, result = set(), set(), []
    for row in mapping['partition']:
        if 'managed_members' not in row:
            continue
        ours, theirs = row['managed_members'], row['ort_members']
        assert not used.intersection(ours) and not native_used.intersection(theirs)
        assert len(ours) == len(set(ours)) == row['managed_nodes']
        assert len(theirs) == len(set(theirs)) == row['ort_nodes']
        used.update(ours); native_used.update(theirs)
        assert all(nodes[n]['calls'] == 60 for n in ours)
        assert all(ort[n]['calls'] == 60 for n in theirs)
        a = sum(nodes[n]['corpus_seconds'] for n in ours)
        b = sum(ort[n]['exclusive_us'] for n in theirs)/3e6
        result.append(dict(group=row['group'], managed_seconds=a, ort_seconds=b, excess_seconds=a-b,
            managed_members=ours, ort_members=theirs))
    assert used == set(nodes) and native_used == set(ort)
    a = managed['phases']['wall']['phase_seconds']['encoder']-sum(r['managed_seconds'] for r in result)
    b = native['phases']['profile']['corpus_phase_seconds']['encoder']-sum(r['ort_seconds'] for r in result)
    assert min(a,b) >= 0
    result.append(dict(group='Encoder outside timed operators', managed_seconds=a, ort_seconds=b, excess_seconds=a-b))
    for name in ['frontend','decoder']:
        a = managed['phases']['wall']['phase_seconds'][name]
        b = native['phases']['profile']['corpus_phase_seconds'][name]
        result.append(dict(group=name, managed_seconds=a, ort_seconds=b, excess_seconds=a-b))
    a = managed['phases']['wall']['remainder_seconds']
    b = native['phases']['profile']['corpus_seconds']-sum(native['phases']['profile']['corpus_phase_seconds'].values())
    assert min(a,b) >= 0
    result.append(dict(group='Outside graph calls', managed_seconds=a, ort_seconds=b, excess_seconds=a-b))
    assert math.isclose(sum(r['managed_seconds'] for r in result), managed['corpus']['wall'], abs_tol=1e-11)
    assert math.isclose(sum(r['ort_seconds'] for r in result), native['phases']['profile']['corpus_seconds'], abs_tol=1e-11)
    return result
