"""Split fresh projection clocks using exact edges, weights and native shapes."""
import collections
import math
from pathlib import Path
from partition import ROOT, TOOLS, pin, read


def main():
    gap = ROOT/'artifacts/parakeet-decoder-lstm-layout-gap-20260927'
    assert pin(gap/'closed.json')['sha256'] == 'f48dec279942297c7561d38391e6ccc66f6780eb8bda630acba2a7af0b849461'
    proof = read(gap/'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(gap/'analysis.json')
    value = read(gap/'analysis.json')
    profiles = {}
    for role, folder in [('managed', 'profile'), ('native', 'ort-profile')]:
        base = ROOT/f'artifacts/parakeet-decoder-lstm-layout-{folder}-amd-20260927'
        assert pin(base/'closed.json') == value[role+'_profile']
        closed = read(base/'closed.json')
        assert closed['passed'] and closed['analysis'] == pin(base/'analysis.json')
        profiles[role] = read(base/'analysis.json')
    group = value['partition'][0]
    assert group['group'] == 'Constant projections including scale'
    managed = {r['name']:r for r in profiles['managed']['phases']['wall']['node_rows'] if r['graph']=='encoder'}
    native = profiles['native']['profiles']['encoder']
    clocks = {r['name']:r for r in native['node_clocks']}
    records, used = [], set()
    for name in group['ort_members']:
        fused = name.endswith('/MatMulScaleFusion/')
        original = name.removesuffix('/MatMulScaleFusion/')
        row = managed[original]
        assert row['op']=='MatMul' and row['calls']==clocks[name]['calls']==60
        weight = row['constant_inputs'][1]
        assert weight and weight['type']=='Float' and len(weight['dims'])==2
        members = [original]
        if fused:
            scales = [n for n in group['managed_members'] if managed[n]['op']=='Mul'
                and row['outputs'][0] in managed[n]['inputs']]
            assert len(scales)==1
            members += scales
        assert not used.intersection(members)
        used.update(members)
        shapes = [r for r in native['shapes'] if r['name']==name]
        assert sum(r['calls'] for r in shapes)==80
        sizes = set()
        for shape in shapes:
            assert len(shape['inputs'])==len(shape['outputs'])==1
            a = shape['inputs'][0]['float']; c = shape['outputs'][0]['float']
            assert a[-1]==weight['dims'][0] and c[-1]==weight['dims'][1]
            assert a[:-1]==c[:-1] and math.prod(a[:-2])==1
            sizes.add(a[-2])
        if '/feed_forward' in original:
            family = 'Feed-forward expansion' if weight['dims']==[1024,4096] else 'Feed-forward contraction including scale'
            assert weight['dims'] in [[1024,4096],[4096,1024]]
        elif '/self_attn/' in original:
            assert weight['dims']==[1024,1024]
            family = original.split('/self_attn/')[1].split('/')[0]
        else:
            assert original=='/pre_encode/out/MatMul' and weight['dims']==[4096,1024]
            family = 'Stem output projection'
        records.append(dict(group=family, managed_members=members, native=name,
            weight=weight, rows=sorted(sizes), native_runtime_B_omitted=True,
            managed_seconds=sum(managed[n]['corpus_seconds'] for n in members),
            ort_seconds=clocks[name]['exclusive_us']/3e6))
    assert len(records)==217 and used==set(group['managed_members']) and len(used)==265
    rows=[]
    for family in dict.fromkeys(r['group'] for r in records):
        selected=[r for r in records if r['group']==family]
        a=sum(r['managed_seconds'] for r in selected); b=sum(r['ort_seconds'] for r in selected)
        rows.append(dict(group=family, matrices=len(selected), managed_seconds=a, ort_seconds=b, difference=a-b))
    assert math.isclose(sum(r['managed_seconds'] for r in rows),group['managed_seconds'],abs_tol=1e-11)
    assert math.isclose(sum(r['ort_seconds'] for r in rows),group['ort_seconds'],abs_tol=1e-11)
    import json
    path=TOOLS/'projection-breakdown-20260927.json'
    with path.open('x',encoding='utf8') as stream:
        json.dump(dict(passed=True, inference_calls=0, source=pin(Path(__file__)),
            partition=pin(gap/'closed.json'), groups=rows, records=records),stream,indent=2)
    print(json.dumps(dict(passed=True, groups=rows)))


if __name__=='__main__':
    main()
