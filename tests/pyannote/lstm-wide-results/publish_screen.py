"""Publish every retained M25 clock and independently aggregate exact tick fractions."""
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-lstm-wide-screen-amd-20260923'
OUT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf8'))
def pin(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

closed=BASE/'closed.json';assert pin(closed)['sha256']=='0cde23e5719eac520b57e6123e5caa76cbed3a586948446b13b8262479a1c3db'
proof=read(closed);assert proof['passed']
for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
analysis=read(BASE/'analysis.json');capture=read(BASE/'collected/references/capture.json')
clocks=[];preps=[];aggregates={}
for name in ['selected-qualify','candidate-qualify','selected-0','candidate-1','candidate-2','selected-3']:
    events=[json.loads(line) for line in (BASE/'collected'/name/'events.jsonl').read_text().splitlines()]
    case_times=[[] for _ in range(12)]
    for event in events:
        if event['kind'] not in ['prepare','call']:continue
        seconds=Fraction(event['end']-event['start'],event['frequency']);source=capture['calls'][event['ordinal']]
        row=dict(process=name,case=source['name'],node=source['index'],ordinal=event['ordinal'],start=event['start'],end=event['end'],frequency=event['frequency'],seconds=float(seconds))
        if event['kind']=='prepare':preps.append(row)
        else:
            row.update(pass_index=event['pass'],repeat=event['repeat'],qualification=name.endswith('-qualify'),warmup=event['pass']==-1)
            clocks.append(row)
            if not row['qualification'] and not row['warmup']:case_times[event['ordinal']].append(seconds)
    if not name.endswith('-qualify'):
        means=[sum(v)/len(v) for v in case_times];aggregate=sum(means);aggregates[name]=aggregate
        assert all(math.isclose(float(a),b,rel_tol=1e-14) for a,b in zip(means,analysis['results'][name]['means']))
        assert math.isclose(float(aggregate),analysis['decision']['processes'][name]['aggregate'],rel_tol=1e-14)
assert len(clocks)==2400 and len(preps)==72
assert sum(r['warmup'] for r in clocks)==588
assert sum(not r['warmup'] and not r['qualification'] for r in clocks)==1764
selected=(aggregates['selected-0']+aggregates['selected-3'])/2
candidate=(aggregates['candidate-1']+aggregates['candidate-2'])/2
ratio=float(candidate/selected);gate=analysis['decision']['gates'][0]
assert math.isclose(ratio,gate['ratio'],rel_tol=1e-14)
for stem,rows in [('clocks',clocks),('preparations',preps)]:
    with (OUT/(stem+'-20260923.csv')).open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(passed=True,clocks=len(clocks),preparations=len(preps),selected=float(selected),candidate=float(candidate),ratio=ratio,improvement_percent=100*(1-ratio),resources=sum(r['samples'] for r in analysis['resources']),peak_rss=max(r['peak_rss'] for r in analysis['resources']))))
