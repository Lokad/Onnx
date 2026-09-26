"""Reconcile complete request time and the exact 72 activation groups."""
from collections import Counter
import json
import math
from pathlib import Path
from run import ROOT, BASE, pin, read, write

MAPPING = ROOT / 'artifacts/parakeet-packed-final-row-gap-20260925'
RESULT = ROOT / 'artifacts/parakeet-pad-current-gap-20260927'
OUT = ROOT / 'tests/parakeet/pad-current-profile-results'


def descriptors_equal(before, after):
    old = {(r['graph'], r['id']): r for r in before}
    new = {(r['graph'], r['id']): r for r in after}
    assert len(old) == len(before) and len(new) == len(after), 'Duplicate graph node'
    assert old.keys() == new.keys()
    def descriptor(row):
        return {k: v for k, v in row.items() if k not in ['ticks', 'corpus_seconds']}
    for key in old:
        assert descriptor(old[key]) == descriptor(new[key]), key


def partition_reports(managed, mapping, native, previous):
    descriptors_equal(previous['phases']['wall']['node_rows'], managed['phases']['wall']['node_rows'])
    nodes = {r['name']: r for r in managed['phases']['wall']['node_rows'] if r['graph'] == 'encoder'}
    ort = {r['name']: r for r in native['profiles']['encoder']['node_clocks']}
    assert len(nodes) == 2856 and len(ort) == 1993
    used, native_used, partition = set(), set(), []
    for row in mapping['partition']:
        if 'managed_members' not in row:
            continue
        ours, theirs = row['managed_members'], row['ort_members']
        assert not used.intersection(ours) and not native_used.intersection(theirs)
        assert len(ours) == len(set(ours)) == row['managed_nodes']
        assert len(theirs) == len(set(theirs)) == row['ort_nodes']
        used.update(ours)
        native_used.update(theirs)
        assert all(nodes[n]['calls'] == 60 for n in ours)
        assert all(ort[n]['calls'] == 60 for n in theirs)
        a = sum(nodes[n]['corpus_seconds'] for n in ours)
        b = sum(ort[n]['exclusive_us'] for n in theirs) / 3e6
        assert math.isclose(b, row['ort_seconds'], abs_tol=1e-12)
        partition.append(dict(group=row['group'], managed_seconds=a, ort_seconds=b, excess_seconds=a-b,
                              managed_members=ours, ort_members=theirs))
    assert used == set(nodes) and native_used == set(ort)
    a = managed['phases']['wall']['phase_seconds']['encoder'] - sum(r['managed_seconds'] for r in partition)
    b = native['phases']['profile']['corpus_phase_seconds']['encoder'] - sum(r['ort_seconds'] for r in partition)
    assert min(a, b) >= 0
    partition.append(dict(group='Encoder outside timed operators', managed_seconds=a, ort_seconds=b, excess_seconds=a-b))
    for name in ['frontend', 'decoder']:
        a = managed['phases']['wall']['phase_seconds'][name]
        b = native['phases']['profile']['corpus_phase_seconds'][name]
        partition.append(dict(group=name, managed_seconds=a, ort_seconds=b, excess_seconds=a-b))
    a = managed['phases']['wall']['remainder_seconds']
    b = native['phases']['profile']['corpus_seconds'] - sum(native['phases']['profile']['corpus_phase_seconds'].values())
    assert min(a, b) >= 0
    partition.append(dict(group='Outside graph calls', managed_seconds=a, ort_seconds=b, excess_seconds=a-b))
    assert math.isclose(sum(r['managed_seconds'] for r in partition), managed['corpus']['wall'], abs_tol=1e-11)
    assert math.isclose(sum(r['ort_seconds'] for r in partition), native['phases']['profile']['corpus_seconds'], abs_tol=1e-11)
    activations, matched = [], set()
    assert len(mapping['activations']) == 72
    for row in mapping['activations']:
        sigmoid, multiply = row['managed_members']
        assert nodes[sigmoid]['op'] == 'Sigmoid' and nodes[multiply]['op'] == 'Mul'
        assert not matched.intersection(row['managed_members'])
        matched.update(row['managed_members'])
        s, m = nodes[sigmoid]['corpus_seconds'], nodes[multiply]['corpus_seconds']
        native_seconds = ort[row['name']]['exclusive_us'] / 3e6
        activations.append(dict(name=row['name'], family=row['family'], managed_members=row['managed_members'],
                                sigmoid_seconds=s, multiply_seconds=m, managed_seconds=s+m,
                                ort_seconds=native_seconds, excess_seconds=s+m-native_seconds))
    assert Counter(r['family'] for r in activations) == {'feed-forward': 48, 'convolution': 24}
    gate, = [r for r in partition if r['group'] == 'Remaining gate sigmoid']
    assert len(gate['managed_members']) == len(gate['ort_members']) == 24
    assert not matched.intersection(gate['managed_members'])
    assert {name for name, row in nodes.items() if row['op'] == 'Sigmoid'} == {
        r['managed_members'][0] for r in activations} | set(gate['managed_members'])
    return partition, activations


def main():
    assert not RESULT.exists() and not OUT.exists()
    proof = read(BASE / 'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(BASE / 'analysis.json')
    for name, wanted in proof['files'].items():
        assert pin(BASE / name) == wanted, name
    managed = read(BASE / 'analysis.json')
    spec = read(BASE / 'bundle/spec.json')
    mapping_proof = read(MAPPING / 'closed.json')
    assert pin(MAPPING / 'closed.json')['sha256'] == '17f0e96862a89606b7e6f56ac5b3ba13fcd36507efa7388bf8c15fff5e0899a7'
    assert mapping_proof['passed'] and mapping_proof['analysis'] == pin(MAPPING / 'analysis.json')
    assert pin(MAPPING / 'analysis.json')['sha256'] == '0faf7f224058af2600067d87e45e3b660f153ad67cc225623ba670f2fc0c7047'
    mapping = read(MAPPING / 'analysis.json')
    values = {}
    for name, wanted in mapping['sources'].items():
        folder = ROOT / 'artifacts' / name
        assert pin(folder / 'closed.json') == wanted['closure'] and pin(folder / 'analysis.json') == wanted['analysis']
        values[name] = read(folder / 'analysis.json')
    previous = values['parakeet-packed-final-row-profile-closure-amd-20260925']
    native = values['parakeet-ort-diagnosis-amd-20260924']
    partition, activations = partition_reports(managed, mapping, native, previous)
    value = dict(passed=True, profile_closure=pin(BASE / 'closed.json'), mapping_closure=pin(MAPPING / 'closed.json'),
        mapping_analysis=pin(MAPPING / 'analysis.json'), native_sources=mapping['sources'],
        partition=partition, activations=activations, corpus=managed['corpus'],
        product=dict(core=spec['core'], data=spec['data']), reference_product=spec['reference_product'],
        phase_over_control=managed['phase_over_control'], wall_over_phase=managed['wall_over_phase'],
        all_graph_descriptors_exact=True, nonoverlapping_complete_partition=True,
        native_profile_is_historical=True, fresh_cross_engine_score=False, overhead_subtracted=False,
        selected_optimization=None, resources=managed['resources'])
    RESULT.mkdir()
    write(RESULT / 'analysis.json', value)
    write(RESULT / 'closed.json', dict(passed=True, analysis=pin(RESULT / 'analysis.json'), reviewer=pin(Path(__file__))))
    OUT.mkdir()
    write(OUT / 'observations-20260927.json', dict(closure=pin(RESULT / 'closed.json'), **value))
    lines = ['# Parakeet after contiguous padding: complete attribution', '',
        f"Actual qualified Core `{spec['core']['sha256']}` and Data `{spec['data']['sha256']}`.",
        'The original runner and reviewed observer are reused. All complete public',
        'results equal the admitted padding product; no product or observer build.', '',
        '| Managed process | Seconds per 20-clip corpus |', '|---|---:|']
    lines += [f'| {name} | {seconds:.6f} |' for name, seconds in managed['corpus'].items()]
    lines += ['', f"Phase/control: {value['phase_over_control']:.6f}; wall/phase: {value['wall_over_phase']:.6f}.",
        'No overhead is subtracted. ORT values are the retained September 24 exact-export',
        'diagnostic profile, not a fresh comparison or release score.', '',
        '| Work | Managed seconds | Dated ORT seconds | Difference |', '|---|---:|---:|---:|']
    lines += [f"| {r['group']} | {r['managed_seconds']:.6f} | {r['ort_seconds']:.6f} | {r['excess_seconds']:.6f} |" for r in partition]
    lines += ['', 'The partition accounts for every encoder node once, all other graph time',
        'and time outside graph calls. Activation details below are a breakdown of',
        'the corresponding partition rows, not additional request time.', '',
        '| Fused activation group | Sigmoid seconds | Separate multiply seconds | ORT fused seconds |', '|---|---:|---:|---:|']
    for family in ['feed-forward', 'convolution']:
        rows = [r for r in activations if r['family'] == family]
        lines.append(f"| {family} ({len(rows)} groups) | {sum(r['sigmoid_seconds'] for r in rows):.6f} | "
                     f"{sum(r['multiply_seconds'] for r in rows):.6f} | {sum(r['ort_seconds'] for r in rows):.6f} |")
    lines += ['', 'The 24 standalone gate Sigmoids remain separately accounted above.',
        'These observations rank causes to investigate; they do not establish an',
        'additive speedup or select another implementation. The rejected vector-exp',
        'screen and isolated Pad repeatability failures retain their verdicts.', '',
        '[Every membership, activation pair, identity and overhead](observations-20260927.json).',
        '[Executed ORT SiLU](../ort-activation-review/diagnosis-20260926.md).',
        '[Matched application comparison](../pad-current-results/application-20260926.md).', '',
        'Closure: `' + pin(RESULT / 'closed.json')['sha256'] + '`.']
    with (OUT / 'diagnosis-20260927.md').open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(passed=True, closure=pin(RESULT / 'closed.json'), corpus=value['corpus'],
                          partition=[{k: v for k, v in r.items() if not k.endswith('_members')} for r in partition])))


if __name__ == '__main__':
    main()
