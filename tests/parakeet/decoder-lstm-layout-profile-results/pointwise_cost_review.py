"""Relate retained arithmetic clocks to the exact full/remainder loop counts.

This descriptive fit estimates costs; it does not measure individual loop times
or establish a candidate's speedup. All measured calls remain included.
"""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-pointwise-cost-amd-20260927'


def pin(path):
    data=path.read_bytes()
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def fit(rows):
    x=np.array([[r['columns']//32,(r['columns']%32)//8,int(r['columns']%8!=0),1] for r in rows],float)
    y=np.array([r['stage_ticks'][4]/r['filters'] for r in rows],float)
    coefficients,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
    assert rank==4
    predicted=x@coefficients
    return dict(calls=len(rows),nanoseconds_per_output_row=coefficients.tolist(),
        r_squared=float(1-np.sum((y-predicted)**2)/np.sum((y-np.mean(y))**2)),
        median_absolute_relative_residual=float(np.median(np.abs(y-predicted)/y)),
        estimated_corpus_seconds=[sum(r['filters']*x[j,i]*coefficients[i] for j,r in enumerate(rows))/3e9 for i in range(4)])


def main():
    assert pin(BASE/'closed.json')['sha256']=='4cd3c5c3ffb4c71c40483544024426b655556b14a33e0c562304a84800e705f0'
    closed=read(BASE/'closed.json')
    assert closed['passed'] and closed['usable_for_candidate_selection']
    paths=['analysis.json','capture-collected/logs/costs.json','capture-collected/probe/observed/result.json']
    for name in paths: assert pin(BASE/name)==closed['files'][name],name
    analysis,costs,public=[read(BASE/n) for n in paths]
    assert analysis['passed'] and analysis['usable_for_candidate_selection'] and costs['frequency']==1000000000
    selected=[]
    for i,record in enumerate(public['records']):
        calls=costs['rows'][i*48:(i+1)*48]
        assert len(calls)==48 and [r['filters'] for r in calls]==[2048,1024]*24
        assert all(record['start_ticks']<=r['start_ticks']<=r['end_ticks']<=record['end_ticks'] for r in calls)
        if record['phase']=='measured': selected.extend(calls)
    assert len(selected)==2880 and len(costs['rows'])==3840
    groups=collections.defaultdict(list)
    for row in selected: groups[row['columns']].append(row)
    shapes=[]
    for width,rows in sorted(groups.items()):
        shapes.append(dict(columns=width,measured_calls=len(rows),full32=width//32,tail8=(width%32)//8,masked=width%8,
            mean_arithmetic_seconds=sum(r['stage_ticks'][4] for r in rows)/len(rows)/1e9))
    totals={name:sum(g['stages'][name] for g in analysis['groups']) for name in costs['stages']}
    totals['remainder']=sum(g['remainder_seconds'] for g in analysis['groups'])
    total=sum(g['corpus_seconds'] for g in analysis['groups'])
    assert abs(sum(totals.values())-total)<1e-12
    result=dict(passed=True,inference_calls=0,closure=pin(BASE/'closed.json'),source=pin(Path(__file__)),
        corpus=analysis['corpus_seconds'],observer_over_control=analysis['observer_over_control'],
        pointwise_seconds=total,stages=totals,arithmetic_fraction=totals['arithmetic']/total,
        all_measured_inputs_and_weights_reused=all(r['input_same'] and r['weight_same'] for r in selected),
        full_column_operation_fraction=sum(r['filters']*(r['columns']//32*32) for r in selected)/sum(r['filters']*r['columns'] for r in selected),
        shapes=shapes,fit_terms=['full32','tail8','masked_tail','fixed'],pooled_fit=fit(selected),
        fits_by_filters={str(m):fit([r for r in selected if r['filters']==m]) for m in [1024,2048]},
        limitation='Fits are descriptive estimates from natural shape variation, not direct loop timers or causal speedup measurements.',
        selected_mechanism='Share AVX2 column-remainder input vectors across eight rows instead of two; retain full panels, packing, reduction and FMA policies.')
    path=TOOLS/'pointwise-cost-observations-20260927.json'
    with path.open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['shapes','fits_by_filters']}))


if __name__=='__main__':main()
