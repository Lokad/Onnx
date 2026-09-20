"""Audit all requests and collection interventions separately from benchmark claims."""
import copy,importlib.util,json
from pathlib import Path
from deploy import BASE,ROOT
from protocol import pin,read,write,validate_records,LIMITS
from memory_protocol import validate_memory

spec=importlib.util.spec_from_file_location('original_audio_resource_audit',ROOT/'tests/audio/amd-comparison/audit.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def main():
    base=BASE/'collected';collection=read(base/'collection.json');state=read(base/'campaign/identity.json')
    assert collection['terminal'] and collection['code']==state['code']==0 and state['complete']
    for name,wanted in collection['files'].items():assert pin(base/name)==wanted,name
    frozen=read(base/'frozen.json');assert frozen['scope']=='whisper-memory-collection' and frozen['collect_after_calls']==[8,16,20]
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    assert state['frozen']==pin(base/'frozen.json') and state['limits']==LIMITS and len(state['runs'])==1
    run=state['runs'][0];assert (run['family'],run['engine'],run['phase'])==('whisper','managed','conformance')
    folder=base/run['output'];value=read(folder/'worker/result.json');manifest=read(base/'manifests/whisper.json')
    validate_records(value,manifest,'conformance');validate_memory(value)
    assert value['runtime']=='.NET 10.0.8' and value['processor_count']==1 and value['flags']=={} and value['engine']=='managed'
    assert value['manifest_sha256']==pin(base/'manifests/whisper.json')['sha256']
    assert value['runner_sha256']==pin(base/'bin/WhisperMemoryCollection.dll')['sha256']
    for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll')]:assert value[key]==manifest[key]==pin(base/'bin'/name)['sha256']
    samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()];resource=old.resource_records(run,samples)
    names={'result.json'}
    for index,row in enumerate(value['records']):
        name=f'{index:03}.json';assert read(folder/'worker'/name)==row;names.add(name)
        assert type(row['allocated_bytes']) is int and row['allocated_bytes']>=0
        if index<16:
            original=read(ROOT/'artifacts/audio-amd-comparison-v2-20260920/collected/campaign/conformance/05-whisper-managed/worker'/name)
            for key in ['name','input_sha256','result']:assert original[key]==row[key],(index,key)
    effects=[]
    for c in value['collections']:
        name=f"collection-{c['after_call']:02}.json";assert read(folder/'worker'/name)==c;names.add(name)
        before=c['before'];after=c['after']
        effects.append(dict(after_call=c['after_call'],seconds=c['seconds'],before=before,after=after,
            managed_reclaimed=before['managed_estimate']-after['managed_estimate'],rss_reduced=before['rss']-after['rss'],
            gen2_collections=after['collections'][2]-before['collections'][2]))
    assert {p.name for p in (folder/'worker').iterdir()}==names
    refusals=0
    for change in [lambda v:v['collections'].pop(),lambda v:v['collections'][0].update(after_call=7),
        lambda v:v['collections'][0].update(inputs_unchanged=False),lambda v:v['collections'][0].update(held_outputs_unchanged=False),
        lambda v:v['records'][1]['memory_before'].update(ticks=0),lambda v:v['records'][0]['memory_after'].update(allocated_total=0),
        lambda v:v['collections'][0]['after'].update(last_gc_index=0),lambda v:v['collections'][0]['after'].update(collections=[0,0,0])]:
        damaged=copy.deepcopy(value);change(damaged);old.refuses(lambda:validate_memory(damaged));refusals+=1
    for change in [lambda v:v['records'].pop(),lambda v:v['records'].reverse(),lambda v:v['records'][0].update(ownership=False)]:
        damaged=copy.deepcopy(value);change(damaged);old.refuses(lambda:validate_records(damaged,manifest,'conformance'));refusals+=1
    for change in [lambda s:s.update(samples=s['samples']+1),lambda s:s.update(peak_rss=-1),lambda s:s.update(code=1)]:
        damaged=copy.deepcopy(run);change(damaged);old.refuses(lambda:old.resource_records(damaged,samples));refusals+=1
    result=dict(passed=True,normal_runtime_qualification=False,benchmark=False,calls=20,original_prefix_bridges=16,interventions=effects,
        resource=resource,records=value['records'],setup_seconds=value['setup_seconds'],runtime=value['runtime'],flags=value['flags'],
        counts=dict(refusals=refusals,resource_samples=len(samples)),births=collection['births'],frozen=pin(base/'frozen.json'))
    write(BASE/'audit.json',result)
    print(json.dumps(dict(passed=True,calls=20,interventions=effects,resource=resource,refusals=refusals)))


if __name__=='__main__':main()
