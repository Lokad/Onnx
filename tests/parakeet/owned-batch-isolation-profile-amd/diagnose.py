"""Refresh the complete partition using the existing exact graph memberships."""
import json
import math
from run import ROOT, BASE, TOOLS, pin, read, write

OLD_GAP = ROOT/'artifacts/parakeet-packed-final-row-gap-20260925'
OUT = TOOLS.parent/'owned-batch-isolation-profile-results'
RESULT = ROOT/'artifacts/parakeet-owned-batch-isolation-gap-20260925'


def descriptors_equal(before, after):
    def by_key(rows):
        result = {(r['graph'],r['id']):r for r in rows}
        assert len(result) == len(rows), 'Duplicate graph node'
        return result
    old, new = by_key(before), by_key(after)
    assert old.keys() == new.keys()
    descriptor = lambda row: {k:v for k,v in row.items() if k not in ['ticks','corpus_seconds']}
    for key in old: assert descriptor(old[key]) == descriptor(new[key]), key


def main():
    assert not RESULT.exists() and not OUT.exists()
    proof = read(BASE/'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(BASE/'analysis.json')
    for name, wanted in proof['files'].items(): assert pin(BASE/name) == wanted, name
    managed = read(BASE/'analysis.json')
    old_proof = read(OLD_GAP/'closed.json')
    assert old_proof['passed'] and old_proof['analysis'] == pin(OLD_GAP/'analysis.json')
    assert pin(OLD_GAP/'analysis.json')['sha256'] == '17f0e96862a89606b7e6f56ac5b3ba13fcd36507efa7388bf8c15fff5e0899a7'
    mapping = read(OLD_GAP/'analysis.json')
    values = {}
    for name, wanted in mapping['sources'].items():
        folder = ROOT/'artifacts'/name
        assert pin(folder/'closed.json') == wanted['closure'] and pin(folder/'analysis.json') == wanted['analysis']
        values[name] = read(folder/'analysis.json')
    old_managed = values['parakeet-packed-final-row-profile-closure-amd-20260925']
    native = values['parakeet-ort-diagnosis-amd-20260924']
    descriptors_equal(old_managed['phases']['wall']['node_rows'], managed['phases']['wall']['node_rows'])
    nodes = {r['name']:r for r in managed['phases']['wall']['node_rows'] if r['graph'] == 'encoder'}
    ort = {r['name']:r for r in native['profiles']['encoder']['node_clocks']}
    assert len(nodes) == 2856 and len(ort) == 1993
    used, native_used, partition = set(), set(), []
    for row in mapping['partition']:
        if 'managed_members' not in row: continue
        ours, theirs = row['managed_members'], row['ort_members']
        assert not used.intersection(ours) and not native_used.intersection(theirs)
        assert len(ours) == len(set(ours)) == row['managed_nodes']
        assert len(theirs) == len(set(theirs)) == row['ort_nodes']
        used.update(ours); native_used.update(theirs)
        assert all(nodes[n]['calls'] == 60 for n in ours)
        assert all(ort[n]['calls'] == 60 for n in theirs)
        a = sum(nodes[n]['corpus_seconds'] for n in ours)
        b = sum(ort[n]['exclusive_us'] for n in theirs)/3e6
        assert math.isclose(b,row['ort_seconds'],abs_tol=1e-12)
        partition.append(dict(group=row['group'],managed_seconds=a,ort_seconds=b,excess_seconds=a-b,
            managed_members=ours,ort_members=theirs))
    assert used == set(nodes) and native_used == set(ort)
    a = managed['phases']['wall']['phase_seconds']['encoder']-sum(r['managed_seconds'] for r in partition)
    b = native['phases']['profile']['corpus_phase_seconds']['encoder']-sum(r['ort_seconds'] for r in partition)
    assert min(a,b) >= 0
    partition.append(dict(group='Encoder outside timed operators',managed_seconds=a,ort_seconds=b,excess_seconds=a-b))
    for name in ['frontend','decoder']:
        a = managed['phases']['wall']['phase_seconds'][name]
        b = native['phases']['profile']['corpus_phase_seconds'][name]
        partition.append(dict(group=name,managed_seconds=a,ort_seconds=b,excess_seconds=a-b))
    a = managed['phases']['wall']['remainder_seconds']
    b = native['phases']['profile']['corpus_seconds']-sum(native['phases']['profile']['corpus_phase_seconds'].values())
    partition.append(dict(group='Outside graph calls',managed_seconds=a,ort_seconds=b,excess_seconds=a-b))
    assert math.isclose(sum(r['managed_seconds'] for r in partition),managed['corpus']['wall'],abs_tol=1e-11)
    assert math.isclose(sum(r['ort_seconds'] for r in partition),native['phases']['profile']['corpus_seconds'],abs_tol=1e-11)
    value = dict(passed=True,profile_closure=pin(BASE/'closed.json'),mapping_closure=pin(OLD_GAP/'closed.json'),
        mapping_analysis=pin(OLD_GAP/'analysis.json'),native_sources=mapping['sources'],partition=partition,
        corpus=managed['corpus'],phase_over_control=managed['phase_over_control'],wall_over_phase=managed['wall_over_phase'],
        all_graph_descriptors_exact=True,nonoverlapping_complete_partition=True,
        native_profile_is_historical=True,fresh_cross_engine_score=False,overhead_subtracted=False,
        selected_optimization=None,resources=managed['resources'])
    RESULT.mkdir(); write(RESULT/'analysis.json',value)
    write(RESULT/'closed.json',dict(passed=True,analysis=pin(RESULT/'analysis.json'),reviewer=pin(TOOLS/'diagnose.py')))
    OUT.mkdir(); write(OUT/'observations-20260925.json',dict(closure=pin(RESULT/'closed.json'),**value))
    lines = ['# Current Parakeet: complete diagnostic attribution','',
        'The current Core e07a4518 / Data 01e9e784 is measured with the unchanged',
        'request runner and reused, compiled-reviewed Data observer. Every complete',
        'public result matches the admitted product exactly. No product was rebuilt.','',
        '| Managed process | Seconds per 20-clip corpus |','|---|---:|']
    lines += [f'| {name} | {seconds:.6f} |' for name,seconds in managed['corpus'].items()]
    lines += ['',f"Phase/control is {value['phase_over_control']:.6f}; wall/phase is {value['wall_over_phase']:.6f}.",
        'No overhead is subtracted. ORT clocks below are the retained September 24',
        'diagnostic profile, not a fresh comparison or a release score. Every node',
        'descriptor matches the reviewed mapping; the partition counts all encoder',
        'nodes once and includes all other graph and request time.','',
        '| Work | Current managed seconds | Dated ORT seconds | Difference |','|---|---:|---:|---:|']
    lines += [f"| {r['group']} | {r['managed_seconds']:.6f} | {r['ort_seconds']:.6f} | {r['excess_seconds']:.6f} |" for r in partition]
    lines += ['', 'These differences rank areas for investigation; they do not establish a',
        'particular mechanism or promise an additive application gain. Inspect the',
        'actual ORT dispatch and matching managed work before selecting one change.',
        'Previous failed padding and activation experiments remain rejected.','',
        '[Complete memberships, clocks, overhead and provenance](observations-20260925.json).',
        '[Fresh release comparison](../owned-batch-isolation-results/release-application-20260925.md).','',
        'Closure: `'+pin(RESULT/'closed.json')['sha256']+'`.']
    (OUT/'diagnosis-20260925.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,closure=pin(RESULT/'closed.json'),corpus=value['corpus'],
        partition=[{k:v for k,v in r.items() if not k.endswith('_members')} for r in partition])))


if __name__ == '__main__': main()
