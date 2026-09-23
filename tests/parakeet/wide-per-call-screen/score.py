"""Exact-rational scoring of the prospectively fixed M39 comparison."""
from fractions import Fraction as F

ORDER=['current-screen0-512','candidate-screen1-512','candidate-screen2-512','current-screen3-512']


def number(value):
    return dict(numerator=value.numerator,denominator=value.denominator,value=float(value))


def evaluate(totals):
    assert list(totals)==ORDER and all(len(v)==12 and all(x>0 for x in v) for v in totals.values())
    controls=[];aggregates={};gates=[];rows=[]
    # First three fixtures are M51 unchanged-path controls; remaining nine are M>=64.
    for name,values in totals.items():
        aggregates[name]=dict(all12=sum(values,F()),wide9=sum(values[3:],F()))
    for role,names in [('current',[ORDER[0],ORDER[3]]),('candidate',ORDER[1:3])]:
        for scope in ['all12','wide9']:
            v=[aggregates[name][scope] for name in names];ratio=max(v)/min(v)
            controls.append(dict(role=role,scope=scope,ratio=number(ratio),limit=1.10,passed=ratio<=F(11,10)))
        for i in range(12):
            v=[totals[name][i] for name in names];ratio=max(v)/min(v)
            controls.append(dict(role=role,scope=f'fixture-{i}',ratio=number(ratio),limit=1.20,passed=ratio<=F(6,5)))
    for i in range(12):
        current=(totals[ORDER[0]][i]+totals[ORDER[3]][i])/2
        candidate=(totals[ORDER[1]][i]+totals[ORDER[2]][i])/2
        ratio=candidate/current
        rows.append(dict(index=i,current=number(current),candidate=number(candidate),ratio=number(ratio),passed=ratio<=F(21,20)))
    scopes={}
    for scope in ['wide9','all12']:
        current=(aggregates[ORDER[0]][scope]+aggregates[ORDER[3]][scope])/2
        candidate=(aggregates[ORDER[1]][scope]+aggregates[ORDER[2]][scope])/2
        scopes[scope]=dict(current=number(current),candidate=number(candidate),ratio=number(candidate/current))
        separated=max(aggregates[name][scope] for name in ORDER[1:3])<min(aggregates[name][scope] for name in [ORDER[0],ORDER[3]])
        gates.append(dict(name='strict-separation-'+scope,passed=separated))
    wide_ratio=F(scopes['wide9']['ratio']['numerator'],scopes['wide9']['ratio']['denominator'])
    gates.append(dict(name='wide-nine-at-least-ten-percent',passed=wide_ratio<=F(9,10)))
    gates.append(dict(name='all-twelve-no-five-percent-regression',passed=all(r['passed'] for r in rows)))
    return dict(admitted=all(x['passed'] for x in [*controls,*gates]),controls=controls,gates=gates,rows=rows,scopes=scopes,
        processes={name:{scope:number(v) for scope,v in sums.items()} for name,sums in aggregates.items()})


def score(reports,capture):
    assert list(reports)==ORDER and len(capture['entries'])==12
    assert [r['m'] for r in capture['entries']]==[51]*3+[106]*3+[167]*3+[225]*3
    totals={}
    for sequence,(name,value) in enumerate(reports.items()):
        assert value['passed'] and value['protocol']=='parakeet-wide-complete-call-60-60-v1'
        assert value['sequence']==sequence and value['role']==name.split('-')[0]
        assert value['calls']==1440 and value['warmups']==value['measured']==720 and len(value['rows'])==12
        frequency=value['frequency'];assert type(frequency) is int and frequency>0
        means=[]
        for i,(row,fixture) in enumerate(zip(value['rows'],capture['entries'],strict=True)):
            assert row['index']==i and [row[k] for k in ['name','node','m','reduction','columns']]==[fixture[k] for k in ['name','node','m','k','n']]
            assert row['exact'] and row['guards'] and row['inputs'] and row['output']==fixture['y']['sha256']
            assert type(row['preparationTicks']) is int and row['preparationTicks']>0
            assert len(row['clocks'])==120
            measured=[]
            for j,clock in enumerate(row['clocks']):
                assert clock['iteration']==j and clock['warmup']==(j<60)
                assert type(clock['ticks']) is int and clock['ticks']>0
                if j>=60:measured.append(F(clock['ticks'],frequency))
            means.append(sum(measured,F())/60)
        totals[name]=means
    return dict(**evaluate(totals),calls=5760,warmups=2880,measured=2880)
