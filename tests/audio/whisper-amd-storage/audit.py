"""Audit four complete fresh timing workers with both unchanged conformance gates."""
import copy
import math
import statistics
from common import *
from storage_contract import STORAGE, validate_storage, verify_schedule
from prepare import conformance_gate


def main():
    audit = original_audit(); base = BASE/'collected'
    receipt = read(base/'collection.json'); assert receipt['terminal'] and receipt['code'] == 0
    for name, expected in receipt['files'].items():
        assert pin(base/name) == expected, name
    frozen = read(base/'frozen.json'); assert frozen['protocol'] == PROTOCOL and frozen['storage'] == STORAGE
    assert frozen['product_source'] == read(BASE/'prepared.json')['product_source']
    for name, expected in frozen['files'].items():
        assert pin(base/name) == expected, name
    assert read(base/'conformance-gate.json') == conformance_gate()
    assert pin(base/'conformance-gate.json') == frozen['prior_conformance']
    state = read(base/'campaign/identity.json'); assert state['complete'] and state['code'] == 0 and 'error' not in state
    assert state['frozen'] == pin(base/'frozen.json') and state['limits'] == LIMITS
    assert 0 < state['seconds'] < LIMITS['campaign_seconds'] and state['prior_conformance'] == frozen['prior_conformance']
    verify_schedule(state['runs']); validate_storage(state['storage'],preflight=True)
    manifest = read(base/'manifests/whisper.json'); observations = []; previous = state['started']; refusals = 0
    for run in state['runs']:
        assert previous <= run['started'] < run['ended']; previous = run['ended']
        validate_storage(run['storage_preflight'],preflight=True)
        assert run['storage_preflight']['free'] == run['preflight_disk'] and run['storage_preflight']['base'] == REMOTE
        folder = base/run['output']; value = read(folder/'worker/result.json')
        validate_records(value,manifest,'timing'); audit.worker_identity(value,manifest,frozen,base,run['engine'])
        assert run['command'][0] == (frozen['dotnet'] if run['engine'] == 'managed' else 'python3')
        assert run['command'][2 if run['engine']=='managed' else 3:] == [REMOTE+'/assets',REMOTE+'/manifests/whisper.json',REMOTE+'/'+run['output']+'/worker','timing']
        samples = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        resource = audit.resource_records(run,samples)
        for sample in samples:
            validate_storage(sample['storage']); assert sample['storage']['free'] == sample['disk']
            for field in ['base','mount','filesystem','device','mount_device','root_device']:
                assert sample['storage'][field] == state['storage'][field]
        for i,row in enumerate(value['records']):
            assert read(folder/'worker'/f'{i:03}.json') == row
        for change in [lambda v:v['records'].pop(),lambda v:v['records'].reverse(),lambda v:v['records'][0].update(ownership=False)]:
            damaged=copy.deepcopy(value);change(damaged);audit.refuses(lambda:validate_records(damaged,manifest,'timing'));refusals+=1
        for change in [lambda v:v.update(runner_sha256='0'*64),lambda v:v.update(manifest_sha256='0'*64),lambda v:v.update(engine='invalid')]:
            damaged=copy.deepcopy(value);change(damaged);audit.refuses(lambda:audit.worker_identity(damaged,manifest,frozen,base,run['engine']));refusals+=1
        for change in [lambda r:r.update(samples=r['samples']+1),lambda r:r.update(peak_rss=-1),lambda r:r.update(code=1)]:
            damaged=copy.deepcopy(run);change(damaged);audit.refuses(lambda:audit.resource_records(damaged,samples));refusals+=1
        for change in [lambda r:r.update(free=STORAGE['remaining']-1),lambda r:r.update(device=r['root_device'])]:
            damaged=copy.deepcopy(samples[0]['storage']);change(damaged);audit.refuses(lambda:validate_storage(damaged));refusals+=1
        observations.append(dict(name=run['name'],phase='timing',family='whisper',engine=run['engine'],resource=resource,value=value,
            min_root_free=min(s['storage']['root_free'] for s in samples)))
    assert previous <= state['ended']; table = []
    for name,cases in [('complete-corpus',manifest['cases'])]+[(c['name'],[c]) for c in manifest['cases']]:
        names={c['name'] for c in cases}; row=dict(name=name,family='whisper',audio_seconds=sum(c['samples'] for c in cases)/16000)
        for engine in ['managed','ort']:
            visits=[]
            for observed in observations:
                if observed['engine']==engine:
                    totals=[math.fsum(r['seconds'] for r in observed['value']['records'] if r['pass']==p and r['name'] in names) for p in [1,2,3]]
                    visits.append(dict(process=observed['name'],passes=totals,mean=statistics.mean(totals)))
            assert len(visits)==2
            row[engine]=dict(visits=visits,seconds=statistics.mean(v['mean'] for v in visits));row[engine]['rtf']=row[engine]['seconds']/row['audio_seconds']
        row['ratio']=row['managed']['seconds']/row['ort']['seconds'];table.append(row)
    counts=dict(referenced_conformance=40,warmup=0,measured=0)
    for observed in observations:
        for row in observed['value']['records']:counts[row['phase']]+=1
    assert counts==dict(referenced_conformance=40,warmup=80,measured=240) and refusals==44
    write(BASE/'audit.json',dict(passed=True,product_source=frozen['product_source'],table=table,observations=observations,
        counts=counts,refusals=refusals,births=receipt['births'],frozen=pin(base/'frozen.json'),storage=STORAGE))
    print(json.dumps(dict(passed=True,headline=table[0],counts=counts,refusals=refusals)))


if __name__=='__main__':
    main()
