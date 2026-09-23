"""Exact-rational scoring of the prospectively fixed M40 comparison."""
from fractions import Fraction as F

ORDER=['current-screen0-512','candidate-screen1-512','candidate-screen2-512','current-screen3-512']
SHORT=list(range(3))+list(range(12,21))
LONG=list(range(3,12))


def number(value):
    return dict(numerator=value.numerator,denominator=value.denominator,value=float(value))


def evaluate(totals):
    assert list(totals)==ORDER and all(len(v)==21 and all(x>0 for x in v) for v in totals.values())
    controls=[];aggregates={};gates=[];rows=[]
    # Target: original M51 cases plus nine fixed prefixes; original M>=64 cases are controls.
    for name,values in totals.items():
        aggregates[name]=dict(all21=sum(values,F()),short12=sum((values[i] for i in SHORT),F()),unchanged9=sum((values[i] for i in LONG),F()))
    for role,names in [('current',[ORDER[0],ORDER[3]]),('candidate',ORDER[1:3])]:
        for scope in ['all21','short12','unchanged9']:
            v=[aggregates[name][scope] for name in names];ratio=max(v)/min(v)
            controls.append(dict(role=role,scope=scope,ratio=number(ratio),limit=1.10,passed=ratio<=F(11,10)))
        for i in range(21):
            v=[totals[name][i] for name in names];ratio=max(v)/min(v)
            controls.append(dict(role=role,scope=f'fixture-{i}',ratio=number(ratio),limit=1.20,passed=ratio<=F(6,5)))
    for i in range(21):
        current=(totals[ORDER[0]][i]+totals[ORDER[3]][i])/2
        candidate=(totals[ORDER[1]][i]+totals[ORDER[2]][i])/2
        ratio=candidate/current
        rows.append(dict(index=i,current=number(current),candidate=number(candidate),ratio=number(ratio),passed=ratio<=F(21,20)))
    scopes={}
    for scope in ['short12','all21']:
        current=(aggregates[ORDER[0]][scope]+aggregates[ORDER[3]][scope])/2
        candidate=(aggregates[ORDER[1]][scope]+aggregates[ORDER[2]][scope])/2
        scopes[scope]=dict(current=number(current),candidate=number(candidate),ratio=number(candidate/current))
        separated=max(aggregates[name][scope] for name in ORDER[1:3])<min(aggregates[name][scope] for name in [ORDER[0],ORDER[3]])
        gates.append(dict(name='strict-separation-'+scope,passed=separated))
    current=(aggregates[ORDER[0]]['unchanged9']+aggregates[ORDER[3]]['unchanged9'])/2
    candidate=(aggregates[ORDER[1]]['unchanged9']+aggregates[ORDER[2]]['unchanged9'])/2
    scopes['unchanged9']=dict(current=number(current),candidate=number(candidate),ratio=number(candidate/current))
    short_ratio=F(scopes['short12']['ratio']['numerator'],scopes['short12']['ratio']['denominator'])
    gates.append(dict(name='short-twelve-at-least-ten-percent',passed=short_ratio<=F(9,10)))
    gates.append(dict(name='all-twenty-one-no-five-percent-regression',passed=all(r['passed'] for r in rows)))
    return dict(admitted=all(x['passed'] for x in [*controls,*gates]),controls=controls,gates=gates,rows=rows,scopes=scopes,
        processes={name:{scope:number(v) for scope,v in sums.items()} for name,sums in aggregates.items()})


def score(reports,capture):
    assert list(reports)==ORDER and len(capture['entries'])==21
    assert [r['m'] for r in capture['entries']]==[51]*3+[106]*3+[167]*3+[225]*3+[48]*3+[61]*3+[63]*3
    totals={}
    for sequence,(name,value) in enumerate(reports.items()):
        assert value['passed'] and value['protocol']=='parakeet-short-wide-complete-call-60-60-v1'
        assert value['sequence']==sequence and value['role']==name.split('-')[0]
        assert value['calls']==2520 and value['warmups']==value['measured']==1260 and len(value['rows'])==21
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
    return dict(**evaluate(totals),calls=10080,warmups=5040,measured=5040)
