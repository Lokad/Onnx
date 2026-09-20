"""Explicit result selection for the sole declared native recovery attempt."""
import json
from common import pin,read


def directory(base,engine,family):
    if engine=='native' and family=='whisper' and (base/'recovery-plan.json').exists():
        plan=read(base/'recovery-plan.json')
        assert plan['engine']==engine and plan['family']==family
        assert plan['destination']=='process-native-whisper-recovery1'
        assert plan['frozen']==pin(base/'frozen.json')
        return base/plan['destination']
    return base/f'process-{engine}-{family}-run'


def failed_attempt(base):
    if not (base/'recovery-plan.json').exists():return None
    plan=read(base/'recovery-plan.json');frozen=read(base/'frozen.json');limits=frozen['limits']['native']
    assert plan['limits']==limits and plan['frozen']==pin(base/'frozen.json')
    assert plan['cases']==[c['name'] for c in read(base/'manifest.json')['cases']]
    for name,wanted in dict(plan['sources'],**plan['failed_files']).items():assert pin(base/name)==wanted,name
    old=base/'process-native-whisper-run';state=read(old/'identity.json')
    assert state['complete'] is False and state['family']=='whisper' and state['engine']=='native'
    assert state['limits']==limits and state['frozen_sha256']==plan['frozen']['sha256']
    assert 'Available memory limit' in state['error'] and read(old/'complete.json')==dict(code=2)
    samples=[json.loads(s) for s in (old/'samples.jsonl').read_text().splitlines()]
    assert len(samples)==state['samples'] and len(samples)>1
    assert all(s['available']>=limits['available'] for s in samples[:-1]) and samples[-1]['available']<limits['available']
    assert all(0<=s['seconds']<limits['seconds'] for s in samples)
    assert all(a['seconds']<b['seconds'] for a,b in zip(samples,samples[1:]))
    assert max(sum(m['rss'] for m in s['members']) for s in samples)==state['peak_rss']<limits['rss']
    for sample in samples:
        for member in sample['members']:
            assert member['affinity']==[2] and member['birth']==state['members'][str(member['pid'])]
    assert {p.name for p in (old/'worker').iterdir()}=={'00.json'}
    row=read(old/'worker/00.json');assert row['name']=='ES2004a' and row['result']['stop_reason']=='Completed'
    return dict(preserved=True,reason='available memory below fixed guard',completed_cases=[row['name']],
        resource_samples=len(samples),seconds_before_stop=samples[-1]['seconds'],peak_rss=state['peak_rss'],
        last_available=samples[-1]['available'],state=state,completed_record=row,plan=pin(base/'recovery-plan.json'))


def recovery_ready(base):
    if not (base/'recovery-plan.json').exists():return
    plan=read(base/'recovery-plan.json');outcome=read(base/'recovery-outcome.json')
    assert outcome['code']==0 and outcome['child_created'] is True
    state=read(directory(base,'native','whisper')/'identity.json')
    assert state['supervisor']==outcome['supervisor']==read(base/'recovery-supervisor.json')
    assert plan['created']<state['started'] and plan['limits']==state['limits']
    samples=[json.loads(s) for s in (base/'recovery-preflight.jsonl').read_text().splitlines()]
    assert samples and all(a['elapsed']<b['elapsed'] for a,b in zip(samples,samples[1:]))
    last=samples[-1];assert last['stable_seconds']>=plan['stable_preflight_seconds']==60
    assert last['elapsed']<plan['preflight_timeout_seconds']==900
    first=last['elapsed']-last['stable_seconds']
    assert all(s['available']>=plan['limits']['preflight'] for s in samples if s['elapsed']>=first)
