"""Exact complete-call means with all corpus frequencies and fixed gates."""
from fractions import Fraction as F
import math

ORDER=['current-0','candidate-1','candidate-2','current-3']


def number(value):return dict(numerator=value.numerator,denominator=value.denominator,value=float(value))


def evaluate(totals,cases):
    assert list(totals)==ORDER and len(cases)==46
    assert all(len(v)==46 and all(t>0 for t in v) for v in totals.values())
    weighted={name:sum((time*case['weight'] for time,case in zip(values,cases,strict=True)),F()) for name,values in totals.items()}
    controls=[];rows=[]
    for role,names in [('current',[ORDER[0],ORDER[3]]),('candidate',ORDER[1:3])]:
        for i,case in enumerate(cases):
            values=[totals[name][i] for name in names];ratio=max(values)/min(values)
            controls.append(dict(role=role,case=case['name'],ratio=number(ratio),passed=ratio<=F(11,10)))
        ratio=max(weighted[n] for n in names)/min(weighted[n] for n in names)
        controls.append(dict(role=role,case='corpus-weighted',ratio=number(ratio),passed=ratio<=F(11,10)))
    for i,case in enumerate(cases):
        current=(totals[ORDER[0]][i]+totals[ORDER[3]][i])/2
        candidate=(totals[ORDER[1]][i]+totals[ORDER[2]][i])/2
        rows.append(dict(index=i,name=case['name'],current=number(current),candidate=number(candidate),
            ratio=number(candidate/current),passed=candidate/current<=F(21,20)))
    current=(weighted[ORDER[0]]+weighted[ORDER[3]])/2
    candidate=(weighted[ORDER[1]]+weighted[ORDER[2]])/2
    gates=[dict(name='at-least-75-percent-corpus-weighted',passed=candidate/current<=F(1,4)),
        dict(name='strict-process-weighted-separation',passed=max(weighted[n] for n in ORDER[1:3])<min(weighted[n] for n in [ORDER[0],ORDER[3]])),
        dict(name='all-cases-no-five-percent-regression',passed=all(r['passed'] for r in rows))]
    return dict(admitted=all(v['passed'] for v in controls+gates),controls=controls,gates=gates,rows=rows,
        corpus_weighted=dict(current=number(current),candidate=number(candidate),ratio=number(candidate/current)),
        processes={n:number(v) for n,v in weighted.items()})


def score(reports,census):
    assert list(reports)==ORDER
    cases=census['cases'];assert len(cases)==46 and sum(c['weight'] for c in cases)==1920
    batches=sum(c['batch'] for c in cases);totals={};inputs=[];outputs=[]
    for sequence,(name,result) in enumerate(reports.items()):
        assert result['passed'] and result['protocol']=='parakeet-sigmoid-public-rounds-600-180-v1'
        assert result['role']==name.split('-')[0] and result['sequence']==sequence
        assert (result['samples'],result['warmup_samples'],result['measured_samples'])==(35880,27600,8280)
        assert (result['calls'],result['warmups'],result['measured'])==(780*batches,600*batches,180*batches)
        assert len(result['rows'])==46 and type(result['frequency']) is int and result['frequency']>0
        means=[];input_hashes=[];output_hashes=[]
        for index,(row,case) in enumerate(zip(result['rows'],cases,strict=True)):
            assert row['index']==index and all(row[k]==case[k] for k in ['name','shape','kind','dtype','weight','batch'])
            assert row['inputs'] and row['ownership'] and row['setup_ticks']>0
            assert math.isfinite(row['maximum_error']) and 0<=row['maximum_error']<=(1e-6 if case['dtype']=='float' else 1e-12)
            assert len(row['input_sha256'])==len(row['output_sha256'])==64
            input_hashes.append(row['input_sha256']);output_hashes.append(row['output_sha256'])
            assert len(row['clocks'])==780
            measured=[]
            for iteration,c in enumerate(row['clocks']):
                assert c['iteration']==iteration and c['warmup']==(iteration<600)
                assert type(c['ticks']) is int and c['ticks']>0 and type(c['start']) is int and c['start']>0
                if iteration>=600:measured.append(F(c['ticks'],result['frequency']*case['batch']))
            assert len(measured)==180;means.append(sum(measured,F())/180)
        previous=0
        for iteration in range(780):
            for row in result['rows']:
                c=row['clocks'][iteration];assert c['start']>=previous;previous=c['start']+c['ticks']
        totals[name]=means;inputs.append(input_hashes);outputs.append(output_hashes)
    assert all(v==inputs[0] for v in inputs)
    assert outputs[0]==outputs[3] and outputs[1]==outputs[2]
    for index,case in enumerate(cases):
        if case['name'] in ['scalar-option','double','scalar','empty']:
            assert all(v[index]==outputs[0][index] for v in outputs)
    return dict(**evaluate(totals,cases),samples=143520,measured_samples=33120,warmup_samples=110400,
        calls=3120*batches,measured_calls=720*batches,warmup_calls=2400*batches,setups=184,
        input_hashes_equal=True,same_product_output_hashes_equal=True,all_warmups_precede_measures=True)
