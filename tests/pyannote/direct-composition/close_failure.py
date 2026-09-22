"""Preserve the real-caller NaN failure and all successful preparation evidence."""
from prepare import *

state=read(BASE / 'preparation.json');assert state['complete'] and state['code']==1
assert state['runs'][-1]['name']=='caller-normal' and state['runs'][-1]['code']==3762504530
assert all(r['code']==0 for r in state['runs'][:-1])
identities=[state['supervisor']]
samples=0
for row in state['runs']:
    assert row['complete'] and row['seconds']<900
    rows=[json.loads(line) for line in (BASE / 'logs' / (row['name']+'.samples.jsonl')).read_text().splitlines()]
    assert len(rows)==row['samples']>0 and max(r['rss'] for r in rows)==row['peak_rss']
    for r in rows:
        assert r['seconds']<900 and r['rss']<8*1024**3 and r['available']>=1024**3 and r['disk']>=20*1024**3 and r['output_bytes']<=1024**3
        assert r['rss']==sum(p['rss'] for p in r['members'])
        for p in r['members']:assert p['affinity']==[2] and row['members'][str(p['pid'])]==p['birth']
    samples+=len(rows);identities.extend(dict(pid=int(pid),birth=birth) for pid,birth in row['members'].items())
for identity in identities:terminal(identity)
inputs=read(BASE / 'inputs.json');verify(inputs['files']);review()
message='Caller bits rows=32 reduction=64 block=1 groups=2 bias=True pattern=special policy=Auto pass=0 batch=1 index=214 old=ffc00000 new=7fc12345'
assert message in (BASE / 'logs/caller-normal.log').read_text()
assert not (BASE / 'failure-closed.json').exists()
files=dict(inputs['files'])
for folder in [BASE / 'logs',BASE / 'runtime',BASE / 'bridge',BASE / 'caller']:
    files.update({rel(p):pin(p) for p in folder.rglob('*') if p.is_file() and 'obj' not in p.relative_to(folder).parts})
for p in BASE.iterdir():
    if p.is_file():files[rel(p)]=pin(p)
save(BASE / 'failure-closed.json',dict(passed=True,expected_failure=True,files=files,identities=identities,
    resource_samples=samples,failure=message,product_changed=False,model_qualified=False))
print(json.dumps(dict(closed=pin(BASE / 'failure-closed.json'),resources=samples,identities=len(identities))))
