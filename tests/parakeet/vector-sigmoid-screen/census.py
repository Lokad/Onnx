"""All observed encoder Sigmoid shapes and frequencies, plus fixed boundary cases."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def pin(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def read(path):return json.loads(path.read_text(encoding='utf8'))


def census():
    native_base=ROOT/'artifacts/parakeet-ort-diagnosis-amd-20260924'
    gap_base=ROOT/'artifacts/parakeet-packed-final-row-gap-20260925'
    for base in [native_base,gap_base]:
        proof=read(base/'closed.json');assert proof['passed'] and proof['analysis']==pin(base/'analysis.json')
    native=read(native_base/'analysis.json')['profiles']['encoder'];gap=read(gap_base/'analysis.json')
    names={r['name'] for r in gap['activations']}|{r['name'] for r in native['node_clocks'] if r['op']=='Sigmoid'}
    assert len(names)==96
    counter=Counter();nodes=Counter()
    for row in native['shapes']:
        if row['name'] not in names:continue
        assert len(row['inputs'])==len(row['outputs'])==1
        assert row['inputs'][0]==row['outputs'][0] and list(row['inputs'][0])==['float']
        counter[tuple(row['inputs'][0]['float'])]+=row['calls']
        nodes[row['name']]+=row['calls']
    assert set(nodes)==names and all(v==80 for v in nodes.values())
    assert len(counter)==38 and sum(counter.values())==96*80 and all(v%4==0 for v in counter.values())
    cases=[]
    for shape,count in counter.items():
        name=('feed-forward-' if shape[-1]==4096 else 'convolution-')+str(shape[1] if shape[-1]==4096 else shape[-1])
        cases.append(dict(name=name,shape=list(shape),kind='dense',weight=count//4,dtype='float'))
    for name,shape,kind,dtype in [
        ('scalar-option',[1,1024,106],'scalar-option','float'),('double',[8193],'dense','double'),
        ('tail',[17],'dense','float'),('empty',[0,7],'dense','float'),('scalar',[],'dense','float'),
        ('reversed',[33,17],'reversed','float'),('sliced',[33,17],'sliced','float'),('broadcast',[33,17],'broadcast','float')]:
        cases.append(dict(name=name,shape=shape,kind=kind,weight=0,dtype=dtype))
    for index,case in enumerate(cases):
        case['index']=index;case['elements']=math.prod(case['shape'])
        case['batch']=max(1,min(256,65536//max(1,case['elements'])))
    assert len(cases)==46 and len({c['name'] for c in cases})==46 and sum(c['weight'] for c in cases)==1920
    return dict(cases=cases,observed_nodes=96,observed_shapes=38,calls_per_corpus=1920,
        native=pin(native_base/'closed.json'),gap=pin(gap_base/'closed.json'),
        values='Synthetic deterministic (i modulo 257 minus 128)/16; actual runtime shapes and frequencies, not captured activation values.',
        warmup_rounds=600,measured_rounds=180,all_cases_warm_before_measurement=True)


if __name__=='__main__':print(json.dumps(census(),indent=2))
