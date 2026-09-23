"""Record the reviewed ordered reduction loops and helper dispatch before timing."""
import json
from pathlib import Path
from inspect_codegen import ROOT, BASE, OUT, PARSER, parse, pin


def block(row, number):
    return row['body'].split(f'G_M000_IG{number:02}:')[1].split('G_M000_IG')[0]


def main():
    target = Path(__file__).parent/'codegen-review-20260923.json'
    assert not target.exists()
    census = json.loads((OUT/'census.json').read_text())
    assert census['passed'] and census['closure'] == pin(BASE/'closed.json')
    numeric = json.loads((BASE/'analysis.json').read_text())
    assert numeric['numerically_admitted'] and numeric['no_performance_measurement']
    for role, wanted in census['inputs'].items():
        assert pin(BASE/'collected/logs'/(role+'-codegen-512.stdout')) == wanted
    rows = parse(BASE/'collected/logs/candidate-codegen-512.stdout')
    def method(name):
        found = [r for r in rows if ':'+name+'(' in r['method']]
        assert len(found) == 1 and found[0]['tier'] == 'FullOpts'
        return found[0]
    two = method('OrderedWideMultiply2Rows')
    three = method('OrderedWideMultiply3Rows')
    helper = method('RunIsolatedShortWidePackedRows')
    assert (two['bytes'], three['bytes'], helper['bytes']) == (615,739,746)
    # Complete bodies were inspected: block endpoints, row groups and reduction
    # backedges, incoming accumulators, packed-panel addressing and exit stores.
    for row, inner, setup, store, count, step in [(two,12,11,13,8,2),(three,11,10,12,12,3)]:
        body = row['body']
        inner_text = body.split(f'G_M000_IG{inner:02}:')[1].split(f'G_M000_IG{inner+1:02}:')[0]
        assert inner_text.count('vfmadd231ps') == count
        assert f'jl       SHORT G_M000_IG{inner:02}' in inner_text
        assert not any(s in inner_text for s in ['call ', 'ymmword ptr [rbp', 'ymmword ptr [rsp'])
        assert f'add      r13d, {step}' in body
        assert '[r15+0x100]' in body and 'cmp      esi, r13d' in body
        setup_text = body.split(f'G_M000_IG{setup:02}:')[1].split(f'G_M000_IG{inner:02}:')[0]
        store_text = body.split(f'G_M000_IG{store:02}:')[1].split(f'G_M000_IG{store+1:02}:')[0]
        assert setup_text.count('vmovups') == store_text.count('vmovups') == count
    assert all(s in helper['body'] for s in [
        'cmp      eax, 64', 'cmp      r14d, 0x400', 'cmp      r15d, 0x400',
        'test     r15b, 31', 'imul     rdi, rsi', 'cmp      rdi, 0x4000000',
        'OrderedWideMultiply2Rows', 'OrderedWideMultiply3Rows',
        'ShortWideMultiply2Rows', 'ShortWideMultiply3Rows', 'TryPackedAvx512Rows'])
    normal = helper['body'].split('G_M000_IG28:')[1].split('G_M000_IG30:')[0]
    assert normal.index('SharedArrayPool`1[float]:Return') < normal.index('G_M000_IG29:')
    assert normal.index('mov      r8, qword ptr [rbp-0x30]') < normal.index('ShortWideMultiplyRemainder')
    assert 'SharedArrayPool`1[float]:Return' in helper['body'].split('G_M000_IG39:')[1]
    notes = dict(
        traversal='Two-row IG17 and three-row IG17 form each 32-column packed panel. Begin starts at zero; IG08/IG07 computes begin+256 and min(N,end). Each block visits every row group before loading the next begin. Panel advancement follows completion of all reduction blocks.',
        arithmetic='Two-row IG11 loads eight incoming C vectors; IG12 traverses ascending j with four B loads, two broadcasts and eight ordered FMA updates; IG13 stores all eight. Three-row IG10 loads twelve C vectors; IG11 traverses ascending j with three broadcasts, four B loads and twelve ordered FMA updates; IG12 stores all twelve. No vector stack spill, reduction split, scalar tail or horizontal reassociation appears in either new loop.',
        dispatch='The 746-byte FullOpts helper inlines the eligibility guard after the unchanged optional AVX512 branch. Rows>=64, divisibility, positive dimensions>=1024, complete 32-column panels and 64-bit N*K<=67108864 precede the ordered calls. Refusals retain the old two/three-row calls. FMA support folds true on this CPU.',
        ownership='Packing and scratch rental remain outside the guard. IG28 returns scratch before IG29 computes the odd-row pointers and reloads original B. Exceptional finally IG39 also returns scratch. Exact compiled scope and numerical ownership/accumulator checks support these paths.',
        unchanged='All four emitted original packer/arithmetic/remainder instruction streams remain identical after address and local-label normalization.',
        scope='All 26 complete emitted bodies have resolved labels. This separate diagnostic capture establishes the generated mechanism and admission to a fixed screen; it does not establish which tier runs in timed processes or any speedup.')
    files = {p.relative_to(ROOT).as_posix():pin(p) for p in OUT.iterdir() if p.is_file()}
    for path in [Path(__file__),Path(__file__).parent/'inspect_codegen.py',PARSER,BASE/'closed.json',BASE/'analysis.json']:
        files[path.relative_to(ROOT).as_posix()] = pin(path)
    value = dict(passed=True, mechanism_admitted=True, no_performance_measurement=True,
        closure=pin(BASE/'closed.json'), numerical_groups_per_worker=76,
        numerical_values_per_worker=8416305, complete_bodies=len(census['bodies']),
        comparisons=census['comparisons'], kernels=census['kernels'],
        helper=dict(index=helper['index'],bytes=helper['bytes']), notes=notes, files=files)
    target.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(review=pin(target),mechanism_admitted=True,complete_bodies=len(census['bodies']))))


if __name__ == '__main__':
    main()
