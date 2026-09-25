"""Rank retained candidate work against the exact earlier ORT profile; no inference."""
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


def main():
    current_base = ROOT/'artifacts/parakeet-slice-dense-conversion-profile-resume-amd-20260925'
    current = read(current_base/'analysis.json')
    proof = read(current_base/'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(current_base/'analysis.json')
    assert pin(current_base/'closed.json')['sha256'] == '705fad8054d2f8020e99e80b32dca43f8f8c6ab046a63780a65faa062a427990'
    assert current['requests'] == 160 and current['original_request_checks']
    assert current['attribution_only'] and not current['overhead_subtracted']
    earlier_path = ROOT/'tests/parakeet/observed-dense-where-results/current-gap-20260924.json'
    earlier = read(earlier_path)
    assert earlier['passed'] and earlier['no_new_inference']
    sources = {}
    for name, wanted in earlier['sources'].items():
        folder = ROOT/'artifacts'/name
        assert pin(folder/'closed.json') == wanted['closure']
        assert pin(folder/'analysis.json') == wanted['analysis']
        original = read(folder/'closed.json')
        assert original['passed'] and original['analysis'] == wanted['analysis']
        sources[name] = wanted
    old_base = ROOT/'artifacts/parakeet-managed-phase-amd-20260924'
    old = read(old_base/'analysis.json')
    native_base = ROOT/'artifacts/parakeet-ort-diagnosis-amd-20260924'
    native = read(native_base/'analysis.json')
    now = current['phases']['candidate']
    nodes = {r['name']: r for r in now['node_rows'] if r['graph'] == 'encoder'}
    previous = {r['name']: r for r in old['phases']['wall']['node_rows'] if r['graph'] == 'encoder'}
    assert len(nodes) == len(previous) == 2856 and nodes.keys() == previous.keys()
    descriptor = lambda r: {k: v for k, v in r.items() if k not in ['ticks', 'corpus_seconds']}
    for name, row in nodes.items():
        assert descriptor(row) == descriptor(previous[name]) and row['calls'] == 60, name
    ort = {r['name']: r for r in native['profiles']['encoder']['node_clocks']}
    comparison = read(old_base/'comparison.json')
    assert earlier['projection_mapping'] == pin(old_base/'comparison.json')
    assert comparison['managed_closure'] == pin(old_base/'closed.json')
    assert comparison['native_closure'] == pin(native_base/'closed.json')
    used = set(); projections = []; groups = defaultdict(lambda: dict(nodes=0, managed=0., ort=0.))
    for row in comparison['projections']:
        members = row['managed_nodes']
        assert not used.intersection(members)
        used.update(members)
        assert ort[row['ort_name']]['calls'] == 60
        native_seconds = ort[row['ort_name']]['exclusive_us']/3e6
        assert math.isclose(native_seconds, row['ort_seconds'], abs_tol=1e-12)
        actual = sum(nodes[n]['corpus_seconds'] for n in members)
        projections.append(dict(ort_name=row['ort_name'], managed_nodes=members, shape=row['shape'],
            alpha=row['alpha'], managed=actual, ort=native_seconds, excess=actual-native_seconds))
        group = groups[(*row['shape'], row['alpha'])]
        group['nodes'] += 1; group['managed'] += actual; group['ort'] += native_seconds
    assert len(projections) == 217 and len(used) == 265
    conv = read(ROOT/'artifacts/parakeet-convolution-attribution-20260924/analysis.json')
    conv_members = {n for g in conv['modules']['managed']['groups'] for n in g['nodes']}
    native_conv = {n for g in conv['modules']['ort']['groups'] for n in g['nodes']}
    assert len(conv_members) == 489 and len(native_conv) == 265
    padding = read(ROOT/'artifacts/parakeet-observed-dense-where-remaining-padding-20260924/analysis.json')
    pad_members = {r['name'] for r in padding['union_rows']}
    assert len(pad_members) == padding['complete_groups']['candidate']['nodes']
    assert all(nodes[r['name']]['op'] == r['op'] for r in padding['union_rows'])
    families = [dict(family='Constant projections including scale', managed=sum(r['managed'] for r in projections), ort=sum(r['ort'] for r in projections)),
        dict(family='Complete convolution modules', managed=sum(nodes[n]['corpus_seconds'] for n in conv_members), ort=sum(ort[n]['exclusive_us'] for n in native_conv)/3e6),
        dict(family='Encoder padding groups', managed=sum(nodes[n]['corpus_seconds'] for n in pad_members), ort=sum(r['earlier_ort_group']['seconds'] for r in padding['families']))]
    for row in families:
        row['excess'] = row['managed']-row['ort']
    positional = [r for r in projections if any('/self_attn/linear_pos/MatMul' in n for n in r['managed_nodes'])]
    assert len(positional) == 24
    positional_seconds = sum(r['managed'] for r in positional)
    assert math.isclose(positional_seconds, current['positional']['candidate_kernel_seconds'], abs_tol=1e-12)
    phases = [dict(phase=k, managed=v, ort=native['phases']['profile']['corpus_phase_seconds'][k]) for k,v in now['phase_seconds'].items()]
    result = dict(passed=True, no_new_inference=True, no_new_candidate_selected=True,
        candidate_not_yet_integrated=True, ort_profile_is_earlier=True,
        profile_times_not_application_scores=True, overlapping_families_must_not_be_summed=True,
        candidate_profile=dict(closure=pin(current_base/'closed.json'), analysis=pin(current_base/'analysis.json')),
        prior_review=pin(earlier_path), original_sources=sources, reviewer=pin(Path(__file__)),
        encoder_descriptors_checked=2856, phases=phases, families=families,
        positional=dict(nodes=24, managed=positional_seconds, ort=sum(r['ort'] for r in positional)),
        projection_groups=[dict(shape=list(k[:2]),alpha=k[2],**v,excess=v['managed']-v['ort']) for k,v in groups.items()], projections=projections)
    path = OUT/'remaining-gap-20260925.json'
    with path.open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(families=families,phases=phases,positional=result['positional'],projection_groups=result['projection_groups'],report=pin(path))))


if __name__ == '__main__':
    main()
