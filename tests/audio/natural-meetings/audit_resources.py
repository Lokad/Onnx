"""Audit all four recording workers and their retained resource samples."""
from pathlib import Path
import argparse
import copy
import json
import math
import psutil
from common import load,pin,read,write
from evidence import directory as worker_directory,failed_attempt,recovery_ready,profile


def validate(state,samples,frozen,engine,family):
    limits=frozen['limits'][engine]
    assert state['complete'] is True and state['code']==0 and 'error' not in state
    assert state['engine']==engine and state['family']==family and state['mode']=='run'
    assert state['limits']==limits and state['frozen_sha256']==frozen['sha256']
    assert 0<state['seconds']<limits['seconds'] and state['ended']>state['started']
    assert type(state['samples']) is int and state['samples']==len(samples) and samples
    assert state['supervisor']['birth']<=state['child']['birth']
    assert state['members'][str(state['child']['pid'])]==state['child']['birth']
    seen={};previous=-1.;peak=0
    for sample in samples:
        assert math.isfinite(sample['seconds']) and previous<sample['seconds']<state['seconds']
        previous=sample['seconds']
        assert type(sample['available']) is int and sample['available']>=limits['available']
        assert len({m['pid'] for m in sample['members']})==len(sample['members'])
        for member in sample['members']:
            assert type(member['pid']) is int and member['pid']>0
            assert type(member['rss']) is int and member['rss']>=0 and member['affinity']==[2]
            assert member['birth']==state['members'][str(member['pid'])]>=state['child']['birth']
            seen[str(member['pid'])]=member['birth']
        rss=sum(m['rss'] for m in sample['members']);assert rss<limits['rss'];peak=max(peak,rss)
    assert peak==state['peak_rss'] and seen==state['members']
    assert state['accounting']['valid'] is True
    assert math.isfinite(state['accounting']['foreign_cpu_fraction']) and state['accounting']['foreign_cpu_fraction']>=0


def damaged_checks(state,samples,frozen,engine,family):
    limits=frozen['limits'][engine]
    changes=[lambda s,x:s.update(complete=False),lambda s,x:s.update(code=1),
        lambda s,x:s.update(family='wrong'),lambda s,x:s.update(frozen_sha256='0'*64),
        lambda s,x:s.update(samples=len(x)+1),lambda s,x:s.update(seconds=limits['seconds']),
        lambda s,x:s.update(peak_rss=-1),lambda s,x:s['child'].update(birth=s['child']['birth']+1),
        lambda s,x:x[1].update(seconds=x[0]['seconds']),lambda s,x:x[0].update(available=limits['available']-1),
        lambda s,x:x[0]['members'][0].update(rss=-1),lambda s,x:x[0]['members'][0].update(rss=limits['rss']),
        lambda s,x:x[0]['members'][0].update(affinity=[0]),lambda s,x:x[0]['members'][0].update(birth=0),
        lambda s,x:s['accounting'].update(valid=False)]
    assert len(samples)>1 and samples[0]['members']
    for change in changes:
        s,x=copy.deepcopy((state,samples));change(s,x)
        try:validate(s,x,frozen,engine,family)
        except (AssertionError,KeyError,ValueError,TypeError):continue
        raise AssertionError('Damaged resource record accepted')
    return len(changes)


def inspect(base,engine,family,frozen):
    directory=worker_directory(base,engine,family);state=read(directory/'identity.json')
    base,manifest,frozen=profile(base,engine,family)
    samples=[json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
    validate(state,samples,dict(frozen,sha256=pin(base/'frozen.json')['sha256']),engine,family)
    assert read(directory/'complete.json')==dict(code=0)
    account=load('retained_asr_process_accounting',base/'runtime/campaign_processes.py')
    assert account.foreign_fraction(read(directory/'pre.json'),read(directory/'post.json'),state['supervisor']['pid'])==state['accounting']
    result=read(directory/'worker/result.json');assert len(result['records'])==3
    assert {p.name for p in (directory/'worker').iterdir()}=={'00.json','01.json','02.json','result.json'}
    for i,row in enumerate(result['records']):assert row==read(directory/f'worker/{i:02d}.json')
    return dict(engine=engine,family=family,host=manifest['hosts'][engine],seconds=state['seconds'],peak_rss=state['peak_rss'],
        minimum_available=min(s['available'] for s in samples),samples=len(samples),accounting=state['accounting'])


def local_terminal(base):
    births={}
    for path in base.glob('process-*/identity.json'):
        if path.parent.name.endswith('-run') and 'managed' in path.parent.name:continue
        state=read(path)
        for item in [state.get('supervisor'),state.get('child')]:
            if item:births[item['pid']]=item['birth']
        births.update({int(pid):birth for pid,birth in state.get('members',{}).items()})
    item=read(base/'deployment-native.json');births[item['pid']]=item['birth']
    for name in ['recovery-deployment.json','recovery-supervisor.json']:
        if (base/name).exists():
            item=read(base/name);births[item['pid']]=item['birth']
    for pid,birth in births.items():
        try:assert psutil.Process(pid).create_time()!=birth,('Owned process still live',pid,birth)
        except psutil.NoSuchProcess:pass
    return [dict(pid=p,birth=b) for p,b in sorted(births.items())]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=a.artifact.resolve();frozen=read(base/'frozen.json');manifest=read(base/'manifest.json')
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in frozen['native_files'].items():assert pin(Path(name))==wanted,name
    collection=read(base/'collected/collection.json');assert collection['all_owned_processes_terminal'] is True
    assert collection['frozen']==pin(base/'frozen.json') and collection['verified_reusable_files']==frozen['files']
    assert {p.relative_to(base/'collected').as_posix() for p in (base/'collected').rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,wanted in collection['files'].items():assert pin(base/'collected'/name)==wanted,name
    continuation=recovery_ready(base);failed=failed_attempt(base)
    resources=[];refusals=[]
    for engine in ['native','managed']:
        campaign=read(base/('collected' if engine=='managed' else '')/f'campaign-{engine}.json')
        assert campaign==dict(complete=True,outcomes=[dict(family=f,code=2 if engine=='native' and f=='whisper' and failed else 0) for f in manifest['schedule']])
        for family in manifest['schedule']:
            resources.append(inspect(base,engine,family,frozen))
            directory=worker_directory(base,engine,family);state=read(directory/'identity.json')
            samples=[json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
            selected,_,selected_frozen=profile(base,engine,family)
            refusals.append(dict(engine=engine,family=family,count=damaged_checks(state,samples,dict(selected_frozen,sha256=pin(selected/'frozen.json')['sha256']),engine,family)))
    result=dict(passed=True,resources=resources,refusals=refusals,failed_native_attempt=failed,native_terminal_processes=local_terminal(base),
        amd_terminal_processes=collection['terminal_processes'],native_linux_continuation=continuation,frozen=pin(base/'frozen.json'),collection=pin(base/'collected/collection.json'))
    write(a.output,result);print('All four worker resource records pass; damaged records refused',sum(r['count'] for r in refusals))


if __name__=='__main__':main()
