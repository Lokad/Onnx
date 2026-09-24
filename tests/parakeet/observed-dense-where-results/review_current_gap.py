"""Reconcile current retained clocks with proved ORT groups; no model execution."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def closed(folder, digest=None):
    base = ROOT/'artifacts'/folder
    proof = read(base/'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(base/'analysis.json')
    if digest:
        assert pin(base/'closed.json')['sha256'] == digest
    return base, read(base/'analysis.json')


def main():
    current_base, current = closed('parakeet-observed-dense-where-profile-resume-amd-20260924',
        '8a6f509a210641650a9c057f472417dde7b4acc320f86eeb3d34ecbced4cc2e6')
    old_base, old = closed('parakeet-managed-phase-amd-20260924')
    native_base, native = closed('parakeet-ort-diagnosis-amd-20260924',
        '615353b4d8524f8935076357c0949859553c27826d50d53ebd82fb002d146a85')
    conv_base, conv = closed('parakeet-convolution-attribution-20260924',
        '3ddc33493522d9f2fd240fad5d5b610c33e80887c3404f0163b254f6e54de9b3')
    padding_base, padding = closed('parakeet-observed-dense-where-remaining-padding-20260924',
        'c9161db64be4f67061339195862b4680cb9e178c0125adf7ad4f73548e0ce46c')
    root_base, root = closed('parakeet-observed-dense-where-root-amd-v2-20260924',
        'dd612e81a85f74aebe4371ca93e6f17216c6779b927caf66bf400f0710af0f8e')
    assert root['root_source_verified'] and root['inventory']['implementation_flags_equal']
    assert current['requests'] == 160 and current['original_request_checks']
    assert current['attribution_only'] and not current['overhead_subtracted']
    now = current['phases']['candidate']
    nodes = {r['name']: r for r in now['node_rows'] if r['graph'] == 'encoder'}
    previous = {r['name']: r for r in old['phases']['wall']['node_rows'] if r['graph'] == 'encoder'}
    assert len(nodes) == len(previous) == 2856 and nodes.keys() == previous.keys()
    for name, row in nodes.items():
        descriptor = lambda r: {k: v for k, v in r.items() if k not in ['ticks', 'corpus_seconds']}
        assert descriptor(row) == descriptor(previous[name]), name
        assert row['calls'] == 60
    ort = {r['name']: r for r in native['profiles']['encoder']['node_clocks']}
    comparison = read(old_base/'comparison.json')
    assert comparison['managed_closure'] == pin(old_base/'closed.json')
    assert comparison['native_closure'] == pin(native_base/'closed.json')
    projections = []; used = set(); grouped = defaultdict(lambda: dict(nodes=0, managed=0., ort=0.))
    for row in comparison['projections']:
        members = row['managed_nodes']
        assert not used.intersection(members)
        used.update(members)
        assert math.isclose(sum(previous[n]['corpus_seconds'] for n in members), row['managed_seconds'], abs_tol=1e-12)
        native_row = ort[row['ort_name']]
        assert native_row['calls'] == 60
        seconds = native_row['exclusive_us']/3e6
        assert math.isclose(seconds, row['ort_seconds'], abs_tol=1e-12)
        actual = sum(nodes[n]['corpus_seconds'] for n in members)
        projections.append(dict(ort_name=row['ort_name'], managed_nodes=members,
            shape=row['shape'], alpha=row['alpha'], managed=actual, ort=seconds, excess=actual-seconds))
        group = grouped[(*row['shape'], row['alpha'])]
        group['nodes'] += 1; group['managed'] += actual; group['ort'] += seconds
    assert len(projections) == 217 and len(used) == 265
    conv_members = {n for g in conv['modules']['managed']['groups'] for n in g['nodes']}
    native_conv_members = {n for g in conv['modules']['ort']['groups'] for n in g['nodes']}
    assert len(conv_members) == 489 and len(native_conv_members) == 265
    conv_seconds = sum(nodes[n]['corpus_seconds'] for n in sorted(conv_members))
    native_conv_seconds = sum(ort[n]['exclusive_us'] for n in native_conv_members)/3e6
    assert math.isclose(native_conv_seconds, conv['modules']['ort']['seconds'], abs_tol=1e-12)
    totals = [dict(family='Constant projections including scale', managed=sum(r['managed'] for r in projections),
        ort=sum(r['ort'] for r in projections)),
        dict(family='Complete convolution modules', managed=conv_seconds, ort=native_conv_seconds),
        dict(family='All encoder padding groups', managed=padding['complete_groups']['candidate']['seconds'],
             ort=sum(r['earlier_ort_group']['seconds'] for r in padding['families']))]
    for row in totals:
        row['excess'] = row['managed']-row['ort']
    phases = [dict(phase=k, managed=v, ort=native['phases']['profile']['corpus_phase_seconds'][k])
        for k,v in now['phase_seconds'].items()]
    result = dict(passed=True, no_new_inference=True, new_candidate_selected=False,
        ort_profile_is_earlier=True, profile_times_not_application_scores=True,
        overlapping_families_must_not_be_summed=True, release_source='dddb60ef',
        sources={b.name: dict(closure=pin(b/'closed.json'), analysis=pin(b/'analysis.json'))
            for b in [current_base,old_base,native_base,conv_base,padding_base,root_base]},
        projection_mapping=pin(old_base/'comparison.json'), reviewer=pin(Path(__file__)),
        encoder_descriptors_checked=2856, phases=phases, families=totals,
        projection_groups=[dict(shape=list(k[:2]),alpha=k[2],**v,excess=v['managed']-v['ort']) for k,v in grouped.items()],
        projections=projections)
    path = OUT/'current-gap-20260924.json'
    with path.open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(phases=phases,families=totals,report=pin(path))))


if __name__ == '__main__':
    main()
