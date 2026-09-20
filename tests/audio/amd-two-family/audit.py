"""Verify every new timing request and the independently retained conformance gate."""
import copy, importlib.util, json, math, statistics
from pathlib import Path
from stage import BASE,ROOT
from protocol import pin,read,write,LIMITS,validate_records

spec=importlib.util.spec_from_file_location('original_amd_audit',ROOT/'tests/audio/amd-comparison/audit.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
worker_identity=original.worker_identity;resource_records=original.resource_records;refuses=original.refuses


def main():
    base=BASE/'collected';receipt=read(base/'collection.json');assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    frozen=read(base/'frozen.json');assert frozen['scope']==['parakeet','pyannote']
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    prior=ROOT/'artifacts/audio-amd-comparison-v2-20260920/collected';gate=read(base/'prior-gate.json')
    assert gate['passed'] and gate['calls']==48 and gate['scope']==frozen['scope']
    assert pin(prior/'frozen.json')==gate['prior_frozen']==frozen['prior_frozen']
    assert pin(prior/'collection.json')==gate['prior_collection']
    assert pin(base/'prior-failure-closed.json')==gate['failure_closure']
    closure=read(base/'prior-failure-closed.json');assert closure['closure_passed'] and not closure['campaign_passed']
    for name,wanted in closure['files'].items():assert pin(ROOT/name)==wanted,name
    for path,wanted in frozen['external'].items():
        if path.startswith(gate['prior_artifact']+'/'):
            assert pin(prior/path.removeprefix(gate['prior_artifact']+'/'))==wanted,path
    assert gate['workers']==read(prior/'campaign/identity.json')['runs'][:4]
    conformance=[]
    for run in gate['workers']:
        folder=prior/run['output'];manifest=read(base/'manifests'/(run['family']+'.json'))
        assert pin(base/'manifests'/(run['family']+'.json'))==pin(prior/'manifests'/(run['family']+'.json'))
        value=read(folder/'worker/result.json');validate_records(value,manifest,'conformance')
        worker_identity(value,manifest,frozen,base,run['engine'])
        samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        resources=resource_records(run,samples)
        for i,row in enumerate(value['records']):assert read(folder/'worker'/f'{i:03}.json')==row
        conformance.append(dict(name=run['name'],calls=len(value['records']),resource=resources,result=pin(folder/'worker/result.json')))
    assert sum(r['calls'] for r in conformance)==48
    state=read(base/'campaign/identity.json');assert state['complete'] and state['code']==0 and 'error' not in state
    assert state['frozen']==pin(base/'frozen.json') and state['limits']==LIMITS and 0<state['seconds']<LIMITS['campaign_seconds']
    wanted=[('timing',family,engine) for family in ['parakeet','pyannote'] for engine in ['managed','ort','ort','managed']]
    assert [(r['phase'],r['family'],r['engine']) for r in state['runs']]==wanted
    observations=[];refusals=0;previous=state['started']
    for run in state['runs']:
        assert previous<=run['started']<run['ended'];previous=run['ended']
        folder=base/run['output'];manifest=read(base/'manifests'/(run['family']+'.json'));value=read(folder/'worker/result.json')
        validate_records(value,manifest,'timing');worker_identity(value,manifest,frozen,base,run['engine'])
        samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        resources=resource_records(run,samples)
        for i,row in enumerate(value['records']):assert read(folder/'worker'/f'{i:03}.json')==row
        for change in [lambda v:v['records'].pop(),lambda v:v['records'].reverse(),lambda v:v['records'][0].update(ownership=False)]:
            damaged=copy.deepcopy(value);change(damaged);refuses(lambda:validate_records(damaged,manifest,'timing'));refusals+=1
        for change in [lambda v:v.update(runner_sha256='0'*64),lambda v:v.update(manifest_sha256='0'*64),lambda v:v.update(engine='invalid')]:
            damaged=copy.deepcopy(value);change(damaged);refuses(lambda:worker_identity(damaged,manifest,frozen,base,run['engine']));refusals+=1
        for change in [lambda s:s.update(samples=s['samples']+1),lambda s:s.update(peak_rss=-1),lambda s:s.update(code=1)]:
            damaged=copy.deepcopy(run);change(damaged);refuses(lambda:resource_records(damaged,samples));refusals+=1
        observations.append(dict(name=run['name'],phase=run['phase'],family=run['family'],engine=run['engine'],resource=resources,value=value))
    assert previous<=state['ended'];table=[]
    for family in ['parakeet','pyannote']:
        manifest=read(base/'manifests'/(family+'.json'));groups=[(c['name'],[c]) for c in manifest['cases']]
        if family=='parakeet':groups.insert(0,('complete-corpus',manifest['cases']))
        for name,cases in groups:
            row=dict(family=family,name=name,audio_seconds=sum(c['samples'] for c in cases)/16000);names={c['name'] for c in cases}
            for engine in ['managed','ort']:
                visits=[]
                for observation in observations:
                    if (observation['family'],observation['engine'])==(family,engine):
                        totals=[math.fsum(r['seconds'] for r in observation['value']['records'] if r['pass']==p and r['name'] in names) for p in [1,2,3]]
                        visits.append(dict(process=observation['name'],passes=totals,mean=statistics.mean(totals)))
                assert len(visits)==2;row[engine]=dict(visits=visits,seconds=statistics.mean(v['mean'] for v in visits))
                row[engine]['rtf']=row[engine]['seconds']/row['audio_seconds']
            row['ratio']=row['managed']['seconds']/row['ort']['seconds'];table.append(row)
    counts=dict(referenced_conformance=48,timing=sum(len(o['value']['records']) for o in observations),warmup=0,measured=0)
    for o in observations:
        for row in o['value']['records']:counts[row['phase']]+=1
    assert counts==dict(referenced_conformance=48,timing=384,warmup=96,measured=288) and refusals==72
    write(BASE/'audit.json',dict(passed=True,table=table,observations=observations,conformance=conformance,refusals=refusals,counts=counts,births=receipt['births'],frozen=pin(base/'frozen.json')))
    print(json.dumps(dict(passed=True,table=table,refusals=refusals,counts=counts)))


if __name__=='__main__':main()
