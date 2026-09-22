"""Freeze full convolution call forms from the independently closed AMD trace."""
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-blocked-spatial-census-20260922'
NATIVE=ROOT/'artifacts/pyannote-native-layout-amd-20260922'
MANAGED=ROOT/'artifacts/pyannote-selected-profile-amd-20260922'


def read(p):return json.loads(p.read_text(encoding='utf8'))


def pin(p):
    p=Path(p)
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    assert not BASE.exists()
    for folder,sha,relative in [(NATIVE,'aec53d4a19ea336e5cd3c4c631c5aea8e895d42d8da49d04b100999510dfedca',True),
                                (MANAGED,'49dadc65ac0ec6ea63914377857549714f40fed87ca330be823ecc78fe3bd056',False)]:
        assert pin(folder/'closed.json')['sha256']==sha
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin((folder if relative else ROOT)/name)==wanted,name
    graph=read(NATIVE/'embedding-graph-and-execution.json')
    result=read(NATIVE/'collected/profile/result.json')
    trace=NATIVE/'collected/profile'/result['models']['embedding']['profile'];events=read(trace)
    nodes={r['name']:r for r in graph['optimized']['nodes'] if r['domain']=='com.microsoft.nchwc' and r['op']=='Conv'}
    assert len(nodes)==36
    groups=collections.defaultdict(list)
    for index,e in enumerate(events):
        if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time'):
            name=e['name'][:-len('_kernel_time')]
            if name in nodes:groups[name].append((index,e))
    assert set(groups)==set(nodes)
    forms={};calls=[]
    for name,node in nodes.items():
        rows=groups[name];assert len(rows)==6
        shapes=[dict(inputs=e['args']['input_type_shape'],outputs=e['args']['output_type_shape']) for _,e in rows]
        assert all(s==shapes[0] for s in shapes)
        s=shapes[0]
        assert all(set(v)=={'float'} for v in s['inputs']+s['outputs'])
        inputs=[v['float'] for v in s['inputs']];out,=[v['float'] for v in s['outputs']]
        assert len(inputs) in [3,4] and len(inputs[0])==len(inputs[1])==len(out)==4
        x,w=inputs[:2];attrs=node['attributes'];residual=len(inputs)==4
        assert inputs[2]==[out[1]] and (not residual or inputs[3]==out)
        eligible=(x[0]==1 and attrs['group']==1 and attrs['kernel_shape']==[3,3]
            and attrs['dilations']==[1,1] and attrs['pads']==[1,1,1,1]
            and attrs['strides'] in [[1,1],[2,2]] and x[1]%16==0 and out[1]%16==0 and out[1]>=32)
        form=dict(input_shape=x,weight_shape=w,output_shape=out,attributes=attrs,residual=residual,eligible=eligible)
        key=json.dumps(form,sort_keys=True)
        if key not in forms:forms[key]=dict(index=len(forms),**form,multiplicity=0,nodes=[])
        group=forms[key];group['multiplicity']+=1;group['nodes'].append(name)
        calls.append(dict(name=name,form=group['index'],eligible=eligible,profile_event_indices=[i for i,_ in rows]))
    assert sum(v['multiplicity'] for v in forms.values())==36
    BASE.mkdir()
    paths=[Path(__file__).resolve(),NATIVE/'closed.json',MANAGED/'closed.json',trace,
        NATIVE/'embedding-graph-and-execution.json',ROOT/'.agent/m13-pyannote-blocked-spatial-20260922.md',
        ROOT/'src/Lokad.Onnx/TensorOps.ConvPool.cs',ROOT/'src/Lokad.Onnx/Zzz.ConvDirectOutput.cs',ROOT/'src/Lokad.Onnx/Zzz.ConvDirectOutputKernel.cs',
        ROOT/'artifacts/pyannote-two-column-20260922/payload/shapes.json']
    files={p.relative_to(ROOT).as_posix():pin(p) for p in paths}
    # Snapshot the living plan so later progress updates do not invalidate the
    # prospective declaration retained for this census.
    plan=ROOT/'.agent/m13-pyannote-blocked-spatial-20260922.md'
    (BASE/'prospective-plan.md').write_bytes(plan.read_bytes());files.pop(plan.relative_to(ROOT).as_posix())
    files[(BASE/'prospective-plan.md').relative_to(ROOT).as_posix()]=pin(BASE/'prospective-plan.md')
    value=dict(passed=True,full_convolution_forms=len(forms),calls=36,profile_events=sum(len(v) for v in groups.values()),
        eligible_calls=sum(r['eligible'] for r in calls),fallback_calls=sum(not r['eligible'] for r in calls),
        forms=list(forms.values()),node_calls=calls,
        scope='Shape/call-frequency census only. Native diagnostic durations are not component timings. Earlier 22 matrix tile shapes remain a separate regression inventory.')
    save(BASE/'census.json',value)
    files[(BASE/'census.json').relative_to(ROOT).as_posix()]=pin(BASE/'census.json')
    save(BASE/'closed.json',dict(passed=True,files=files,census=pin(BASE/'census.json'),no_inference_or_timing=True))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),forms=value['full_convolution_forms'],calls=36,
        eligible=value['eligible_calls'],fallback=value['fallback_calls'],events=value['profile_events'])))


if __name__=='__main__':main()
