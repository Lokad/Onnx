"""Audit every integrated Whisper request, process identity and resource sample."""
import copy,importlib.util,json,math,statistics
from common import *
sys.path.insert(0,str(LOCAL_SITE))
spec=importlib.util.spec_from_file_location('original_audio_audit',ROOT/'tests/audio/amd-comparison/audit.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)

def main():
    base=BASE/'collected';receipt=read(base/'collection.json');assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    frozen=read(base/'frozen.json');assert frozen['scope']==['whisper'] and frozen['product_source']==read(BASE/'prepared.json')['product_source']
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    assert frozen['managed_runtime']==read(CONTRACTS/'frozen.json')['managed_runtime']
    assert frozen['dotnet']==frozen['managed_runtime']['host']
    for name,wanted in frozen['managed_runtime']['files'].items():assert frozen['external'][name]==wanted,name
    prior=PRIOR/'collected';gate=read(base/'prior-native-gate.json');old=read(prior/'frozen.json')
    assert gate['passed'] and gate['calls']==20 and pin(base/'prior-native-gate.json')==frozen['prior_native']
    assert pin(prior/'frozen.json')==gate['frozen'] and pin(PRIOR/'failure-closed.json')==gate['closure']
    assert read(PRIOR/'failure-closed.json')['closure_passed']
    prior_run=gate['worker'];assert prior_run==read(prior/'campaign/identity.json')['runs'][4]
    old_manifest=read(prior/'manifests/whisper.json');manifest=read(base/'manifests/whisper.json')
    assert pin(prior/'manifests/whisper.json')==gate['manifest']
    assert {k:v for k,v in old_manifest.items() if k not in ['product_source','core_sha256','data_sha256']}=={k:v for k,v in manifest.items() if k not in ['product_source','core_sha256','data_sha256']}
    for name,wanted in frozen['external'].items():
        if name.startswith(PRIOR_REMOTE+'/'):assert pin(prior/name.removeprefix(PRIOR_REMOTE+'/'))==wanted,name
    folder=prior/prior_run['output'];value=read(folder/'worker/result.json');assert pin(folder/'worker/result.json')==gate['result']
    validate_records(value,old_manifest,'conformance');original.worker_identity(value,old_manifest,old,prior,'ort')
    resources=original.resource_records(prior_run,[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]);assert resources==gate['resource']
    for i,row in enumerate(value['records']):assert read(folder/'worker'/f'{i:03}.json')==row
    state=read(base/'campaign/identity.json');assert state['complete'] and state['code']==0 and 'error' not in state
    assert state['frozen']==pin(base/'frozen.json') and state['limits']==LIMITS and 0<state['seconds']<LIMITS['campaign_seconds']
    assert [(r['phase'],r['family'],r['engine']) for r in state['runs']]==[('conformance','whisper','managed')]+[('timing','whisper',e) for e in ['managed','ort','ort','managed']]
    current_gate=read(base/'campaign/conformance-gate.json');snapshot=base/'campaign/conformance-identity.json'
    assert current_gate['passed'] and current_gate['identity']==pin(snapshot) and current_gate['frozen']==state['frozen']
    assert read(snapshot)['runs']==state['runs'][:1]
    observations=[];refusals=0;previous=state['started']
    for run in state['runs']:
        assert previous<=run['started']<run['ended'];previous=run['ended'];folder=base/run['output'];value=read(folder/'worker/result.json')
        validate_records(value,manifest,run['phase']);original.worker_identity(value,manifest,frozen,base,run['engine'])
        if run['engine']=='managed':assert run['command'][0]==frozen['dotnet']
        samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()];resources=original.resource_records(run,samples)
        for i,row in enumerate(value['records']):assert read(folder/'worker'/f'{i:03}.json')==row
        if run['phase']=='conformance':assert current_gate['workers']=={run['name']:pin(folder/'worker/result.json')}
        for change in [lambda v:v['records'].pop(),lambda v:v['records'].reverse(),lambda v:v['records'][0].update(ownership=False)]:
            damaged=copy.deepcopy(value);change(damaged);original.refuses(lambda:validate_records(damaged,manifest,run['phase']));refusals+=1
        for change in [lambda v:v.update(runner_sha256='0'*64),lambda v:v.update(manifest_sha256='0'*64),lambda v:v.update(engine='invalid')]:
            damaged=copy.deepcopy(value);change(damaged);original.refuses(lambda:original.worker_identity(damaged,manifest,frozen,base,run['engine']));refusals+=1
        for change in [lambda r:r.update(samples=r['samples']+1),lambda r:r.update(peak_rss=-1),lambda r:r.update(code=1)]:
            damaged=copy.deepcopy(run);change(damaged);original.refuses(lambda:original.resource_records(damaged,samples));refusals+=1
        observations.append(dict(name=run['name'],phase=run['phase'],family='whisper',engine=run['engine'],resource=resources,value=value))
    assert previous<=state['ended'];table=[]
    for name,cases in [('complete-corpus',manifest['cases'])]+[(c['name'],[c]) for c in manifest['cases']]:
        names={c['name'] for c in cases};row=dict(name=name,family='whisper',audio_seconds=sum(c['samples'] for c in cases)/16000)
        for engine in ['managed','ort']:
            visits=[]
            for o in observations:
                if (o['phase'],o['engine'])==('timing',engine):
                    totals=[math.fsum(r['seconds'] for r in o['value']['records'] if r['pass']==p and r['name'] in names) for p in [1,2,3]]
                    visits.append(dict(process=o['name'],passes=totals,mean=statistics.mean(totals)))
            assert len(visits)==2;row[engine]=dict(visits=visits,seconds=statistics.mean(v['mean'] for v in visits));row[engine]['rtf']=row[engine]['seconds']/row['audio_seconds']
        row['ratio']=row['managed']['seconds']/row['ort']['seconds'];table.append(row)
    counts=dict(referenced_native_conformance=20,managed_conformance=0,timing=0,warmup=0,measured=0)
    for o in observations:
        if o['phase']=='conformance':counts['managed_conformance']+=len(o['value']['records'])
        else:
            counts['timing']+=len(o['value']['records'])
            for r in o['value']['records']:counts[r['phase']]+=1
    assert counts==dict(referenced_native_conformance=20,managed_conformance=20,timing=320,warmup=80,measured=240) and refusals==45
    write(BASE/'audit.json',dict(passed=True,product_source=frozen['product_source'],table=table,observations=observations,prior_native=gate,counts=counts,refusals=refusals,births=receipt['births'],frozen=pin(base/'frozen.json')))
    print(json.dumps(dict(passed=True,headline=table[0],counts=counts,refusals=refusals)))

if __name__=='__main__':main()
