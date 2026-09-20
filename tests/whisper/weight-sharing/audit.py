"""Audit every completed request, original allocation bridge and resource record."""
import copy,importlib.util,json,statistics,sys
from pathlib import Path
from deploy import BASE,ROOT
sys.path.append(str(ROOT/'tests/whisper/memory-collection'))
sys.path.append(str(ROOT/'tests/whisper/buffer-reuse'))
from protocol import pin,read,write,validate_records,LIMITS
from reuse_protocol import validate_reuse,allocation_gate
from sharing_protocol import validate_sharing

spec=importlib.util.spec_from_file_location('original_audio_resource_audit',ROOT/'tests/audio/amd-comparison/audit.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def main():
    base=BASE/'collected';receipt=read(base/'collection.json');state=read(base/'campaign/identity.json');frozen=read(base/'frozen.json')
    assert receipt['terminal'] and receipt['code']==state['code']==0 and state['complete'] and 'error' not in state
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    assert state['frozen']==pin(base/'frozen.json')==pin(BASE/'frozen.json') and state['limits']==LIMITS
    assert frozen['scope']=='whisper-weight-sharing' and frozen['explicit_gc'] is False
    assert frozen['conformance_calls']==20 and frozen['endurance_calls']==80
    assert 0<state['seconds']<LIMITS['campaign_seconds']
    assert [(r['family'],r['engine'],r['phase']) for r in state['runs']]==[('whisper','managed','conformance'),('whisper','managed','timing')]
    manifest=read(base/'manifests/whisper.json');original_manifest=read(ROOT/'artifacts/audio-amd-comparison-v2-20260920/collected/manifests/whisper.json')
    for key in manifest:
        if key not in ['core_sha256','data_sha256']:assert manifest[key]==original_manifest[key],key
    original=[read(base/'original-prefix'/f'{i:03}.json') for i in range(16)]
    prefix=ROOT/'artifacts/audio-amd-comparison-v2-20260920/collected/campaign/conformance/05-whisper-managed/worker'
    for i in range(16):assert pin(base/'original-prefix'/f'{i:03}.json')==pin(prefix/f'{i:03}.json')
    gate=read(base/'campaign/conformance-gate.json');observations=[];previous=state['started'];refusals=0
    for run in state['runs']:
        assert previous<=run['started']<run['ended'];previous=run['ended']
        folder=base/run['output'];value=read(folder/'worker/result.json')
        validate_records(value,manifest,run['phase']);validate_reuse(value);validate_sharing(value)
        assert value['engine']=='managed' and value['runtime']=='.NET 10.0.8' and value['processor_count']==1 and value['flags']=={}
        assert value['manifest_sha256']==pin(base/'manifests/whisper.json')['sha256']
        assert value['runner_sha256']==pin(base/'bin/WhisperWeightSharing.dll')['sha256']
        for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll')]:assert value[key]==manifest[key]==pin(base/'bin'/name)['sha256']
        samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        resource=old.resource_records(run,samples)
        for i,row in enumerate(value['records']):assert read(folder/'worker'/f'{i:03}.json')==row
        assert {p.name for p in (folder/'worker').iterdir()}=={'result.json'}|{f'{i:03}.json' for i in range(len(value['records']))}
        if run['phase']=='conformance':
            expected=allocation_gate(value,original);expected.update(worker=pin(folder/'worker/result.json'),frozen=pin(base/'frozen.json'));assert gate==expected
            for damage in [lambda v:v['records'][1].update(allocated_bytes=gate['original_allocated_bytes']),
                lambda v:v['records'][0].update(input_sha256='0'*64)]:
                bad=copy.deepcopy(value);damage(bad);old.refuses(lambda:allocation_gate(bad,original));refusals+=1
        for damage in [lambda v:v['records'].pop(),lambda v:v['records'].reverse(),lambda v:v['records'][0].update(ownership=False)]:
            bad=copy.deepcopy(value);damage(bad);old.refuses(lambda:validate_records(bad,manifest,run['phase']));refusals+=1
        for damage in [lambda v:v['records'][1]['pools']['encodingExecution'].update(allocated_new_bytes=16*1024**2+1),
            lambda v:v['records'][0]['pools']['encodingExecution'].update(cache_bytes=512*1024**2+1),
            lambda v:v['records'][0]['pools']['pastExecution'].update(cache_count=257),
            lambda v:v['records'][1]['memory_before'].update(ticks=0),
            lambda v:v['records'][0].update(gc_after=[-1,0,0])]:
            bad=copy.deepcopy(value);damage(bad);old.refuses(lambda:validate_reuse(bad));refusals+=1
        for damage in [lambda v:v.update(samples=v['samples']+1),lambda v:v.update(peak_rss=-1),lambda v:v.update(code=1)]:
            bad=copy.deepcopy(run);damage(bad);old.refuses(lambda:old.resource_records(bad,samples));refusals+=1
        for damage in [lambda v:v['weight_sharing']['after'].update(logical_shared_bytes=1),
            lambda v:v['weight_sharing']['before'].update(shared_payload_bytes=1),
            lambda v:v['weight_sharing']['after']['past']['initializers'][0].update(sha256='0'*64)]:
            bad=copy.deepcopy(value);damage(bad);old.refuses(lambda:validate_sharing(bad));refusals+=1
        allocations=[r['allocated_bytes'] for r in value['records']]
        observations.append(dict(phase=run['phase'],calls=len(allocations),resource=resource,allocated_total=sum(allocations),
            allocated_min=min(allocations),allocated_max=max(allocations),allocated_median=statistics.median(allocations),
            sharing=validate_sharing(value),first_gc=value['records'][0]['gc_before'],last_gc=value['records'][-1]['gc_after'],records=value['records']))
    assert previous<=state['ended']
    result=dict(passed=True,prototype_only=True,benchmark=False,calls=100,gate=gate,observations=observations,
        refusals=refusals,births=receipt['births'],frozen=pin(base/'frozen.json'),collection=pin(base/'collection.json'))
    write(BASE/'audit.json',result)
    print(json.dumps(dict(passed=True,calls=100,gate=gate,resources=[r['resource'] for r in observations],refusals=refusals)))


if __name__=='__main__':main()
