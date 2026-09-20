"""Describe every retained independent-deployment process; never revise its gates."""
from pathlib import Path
import argparse,hashlib,json,statistics as st

RECEIPT='5059cd6ec8cd80b3bafc23a77bcbd78472e0ced0bc8f3b3420cddf5d72f09864'

def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())

def analyze(base):
    assert pin(base/'closed.json')['sha256']==RECEIPT
    closed=json.loads((base/'closed.json').read_text())
    for name,wanted in closed['files'].items():assert pin(base/name)==wanted,name
    folder=base/'collected-aa/result-aa';state=json.loads((folder/'identity.json').read_text())
    assert state['complete'] and state['code']==0 and len(state['runs'])==160
    rows=[]
    for run in state['runs']:
        job=run['job'];path=folder/job['name']/'output/result.json';value=json.loads(path.read_text())
        measured=value['measured'];warm=value['conditioning'];count=len(measured);frequency=value['frequency']
        assert count==48*[32,16,4,4,2][job['case_index']] and count%4==0
        boundaries={}
        for boundary in ['execute','request']:
            values=[r[boundary]/frequency for r in measured]
            quartiles=[st.fmean(values[i*count//4:(i+1)*count//4]) for i in range(4)]
            boundaries[boundary]=dict(mean=st.fmean(values),median=st.median(values),minimum=min(values),maximum=max(values),
                quartile_means=quartiles,last_first=quartiles[-1]/quartiles[0],
                block_means=[st.fmean(values[k*count//48:(k+1)*count//48]) for k in range(48)],
                final_conditioning_quarter=st.fmean(r[boundary]/frequency for r in warm[-max(1,len(warm)//4):]))
        rows.append(dict(job=job,source=pin(path),measured_calls=count,conditioning_calls=len(warm),boundaries=boundaries,
            allocation=dict(minimum=min(r['bytes'] for r in measured),mean=st.fmean(r['bytes'] for r in measured),maximum=max(r['bytes'] for r in measured)),
            gc_counts=[sum(r[g] for r in measured) for g in ['g0','g1','g2']],
            calls_with_gc=sum(any(r[g] for g in ['g0','g1','g2']) for r in measured)))
    cohorts=[]
    for case in ['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']:
        for policy in ['default','memory']:
            group=[r for r in rows if r['job']['case']==case and r['job']['policy']==policy and r['job']['role']!='N']
            assert len(group)==12
            visits=[]
            for visit in range(4):
                members=[r for r in group if r['job']['visit']==visit];assert len(members)==3
                visits.append(dict(visit=visit,spreads={b:{stat:max(r['boundaries'][b][stat] for r in members)/min(r['boundaries'][b][stat] for r in members)
                    for stat in ['mean','median']} for b in ['execute','request']}))
            cohorts.append(dict(case=case,policy=policy,visits=visits,
                measured_drift_range=[min(r['boundaries']['execute']['last_first'] for r in group),max(r['boundaries']['execute']['last_first'] for r in group)]))
    return dict(scope='Post-hoc description of all original samples; no changed verdict, causal attribution, fitted correction or new inference.',
        source_receipt=pin(base/'closed.json'),source_files_verified=len(closed['files']),workers=rows,cohorts=cohorts,
        measured_calls=sum(r['measured_calls'] for r in rows),conditioning_calls=sum(r['conditioning_calls'] for r in rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();v=analyze(a.source.resolve())
    with a.output.open('x',encoding='utf-8') as stream:json.dump(v,stream,indent=2)
    print(json.dumps({k:v[k] for k in ['source_files_verified','measured_calls','conditioning_calls']}))
