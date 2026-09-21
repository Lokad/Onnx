"""Audit native public decisions, actual overlap, every weight and resource sample."""
from pathlib import Path
import copy,importlib.util,json,sys
from common import ROOT,BASE,LOCAL,PRODUCT,PRELUDE,pin,read,write,ssh

sys.path.insert(0,str(ROOT/'tests/whisper/memory-contracts-v2'))
spec=importlib.util.spec_from_file_location('windows_contract_audit',ROOT/'tests/whisper/memory-contracts-v2/audit.py')
local=importlib.util.module_from_spec(spec);spec.loader.exec_module(local)


def application(value,weights,inputs,native,old,short,tokenizer,census):
    result=local.validate_application(value,inputs,native,old,short,tokenizer)
    sharing=local.validate_sharing(dict(weight_sharing=weights));checked=0
    for stage in ['before','after']:
        for name,original in zip(['first','past'],census['graphs'],strict=True):
            entries={r['name']:r for r in weights[stage][name]['initializers']}
            for row in original['rows']:
                actual=entries[row['name']]
                assert actual['bytes']==row['bytes'] and actual['shape']==row['shape'] and actual['sha256']==row['sha256']
                assert actual['type']=={1:'Float',7:'Int64'}[row['type']];checked+=1
    assert checked==1136
    return dict(application=result,sharing=sharing,original_initializer_checks=checked)


def resources(samples,state,limits):
    assert state['limits']==limits and len(samples)==state['samples'] and 0<state['execution_seconds']<state['seconds']<limits['seconds']
    assert state['preflight_available']>=limits['preflight'] and state['preflight_disk']>=limits['preflight_disk']
    previous=0;gaps=[];seen={};threads=0
    for sample in samples:
        assert previous<sample['seconds']<=state['execution_seconds'];gaps.append(sample['seconds']-previous);previous=sample['seconds']
        assert sample['available']>=limits['available'] and sample['disk']>=limits['disk']
        assert sample['members'] and sum(m['rss'] for m in sample['members'])<limits['rss']
        for member in sample['members']:
            assert member['affinity']==[2] and member['rss']>=0 and member['birth']==state['members'][str(member['pid'])]
            assert member['threads'] and all(t['affinity']==[2] and type(t['tid']) is int and t['tid']>0 for t in member['threads'])
            threads+=len(member['threads']);seen[str(member['pid'])]=member['birth']
    gaps.append(state['execution_seconds']-previous);assert max(gaps)<10 and seen==state['members']
    peak=max(sum(m['rss'] for m in s['members']) for s in samples);assert peak==state['peak_rss']
    return dict(samples=len(samples),thread_observations=threads,peak_rss=peak,min_available=min(s['available'] for s in samples),
        min_disk=min(s['disk'] for s in samples),max_gap=max(gaps),execution_seconds=state['execution_seconds'])


def main():
    root=BASE/'collected';receipt=read(root/'collection.json');state=read(root/'run/identity.json');frozen=read(root/'frozen.json')
    assert receipt['terminal'] and state['complete'] and state['code']==0 and 'error' not in state
    assert receipt['frozen']==pin(root/'frozen.json')==pin(BASE/'frozen.json')==state['frozen']
    for name,wanted in receipt['files'].items():assert pin(root/name)==wanted,name
    for name,wanted in frozen['files'].items():assert pin(root/name)==wanted,name
    assert read(BASE/'prepared.json')['local_closure']==pin(LOCAL/'local-closed.json')
    assert read(root/'reference/windows-verification.json')['passed']
    terminal=json.loads(ssh(PRELUDE+'terminal(%r)\nprint(json.dumps(dict(terminal=True)))\n'%receipt['births']));assert terminal['terminal']
    worker=root/'run/worker';value=read(worker/'result.json');weights=read(worker/'weights.json')
    assert pin(worker/'result.json')==state['result'] and value['weights_sha256']==pin(worker/'weights.json')['sha256']
    assert value['runtime']=='.NET 10.0.8' and value['affinity']==4 and value['processor_count']==1 and value['flags']=={}
    for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','WhisperMemoryContractsV2.dll')]:
        assert value[key]==pin(root/'bin'/name)['sha256']==pin(LOCAL/'bin'/name)['sha256']
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:
        assert pin(root/'bin'/name)==pin(PRODUCT/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'/name)
    for stage in ['before','after']:assert read(worker/('weights-'+stage+'.json'))==weights[stage]
    inputs=read(root/'inputs/inputs.json');native=read(root/'reference/native-recording.json');old=read(root/'reference/original-managed.json')
    short=read(root/'short/manifest.json');census=read(root/'reference/weight-census.json')
    assert value['inputs_sha256']==native['inputs_sha256']==pin(root/'inputs/inputs.json')['sha256']
    assert value['short_manifest_sha256']==pin(root/'short/manifest.json')['sha256']
    tokenizer=local.Tokenizer.from_file(str(ROOT/'models/whisper-large-v3-turbo/tokenizer.json'))
    result=application(value,weights,inputs,native,old,short,tokenizer,census)
    for index,row in enumerate(value['cases']):assert read(worker/f"{index:02}-{row['name']}.json")==row
    expected={f"{i:02}-{r['name']}.json" for i,r in enumerate(value['cases'])}|{'result.json','weights.json','weights-before.json','weights-after.json'}
    assert {p.name for p in worker.iterdir()}==expected
    samples=[json.loads(s) for s in (root/'run/samples.jsonl').read_text().splitlines()]
    resource=resources(samples,state,frozen['limits']);preflight=read(root/'run/preflight.json')
    assert preflight['available']==state['preflight_available'] and preflight['disk']==state['preflight_disk']
    damaged=0
    for edit in [lambda v:v.update(refusals=15),lambda v:v['cases'].pop(),lambda v:v['cases'][0].update(ownership=False),
        lambda v:v['cases'][0]['result']['segments'][0].update(text='wrong'),lambda v:v['short_recovery'].update(text='wrong'),
        lambda v:v['concurrent_speech'][1].update(start=v['concurrent_speech'][0]['end']+1,end=v['concurrent_speech'][0]['end']+2),
        lambda v:v['concurrent_speech'][1].update(pcm_sha256='0'*64),lambda v:v['concurrent_speech'][1]['result']['token_ids'].pop(),
        lambda v:v['silent'].update(processed_seconds=599)]:
        bad=copy.deepcopy(value);edit(bad);local.refuses(lambda:application(bad,weights,inputs,native,old,short,tokenizer,census));damaged+=1
    for edit in [lambda v:v['after']['first']['initializers'][0].update(sha256='0'*64),lambda v:v['after'].update(unique_arrays=1),
        lambda v:next(r for r in v['after']['past']['initializers'] if r['name']=='folded:Transpose_801').update(tensor_name='wrong')]:
        bad=copy.deepcopy(weights);edit(bad);local.refuses(lambda:application(value,bad,inputs,native,old,short,tokenizer,census));damaged+=1
    for edit in [lambda v:v[0]['members'][0]['threads'][0].update(affinity=[0]),lambda v:v[0].update(available=0),
        lambda v:v[0]['members'][0].update(birth=0),lambda v:v[0]['members'][0].update(rss=frozen['limits']['rss'])]:
        bad=copy.deepcopy(samples);edit(bad);local.refuses(lambda:resources(bad,state,frozen['limits']));damaged+=1
    result.update(passed=True,private_prototype=True,benchmark=False,resources=resource,damaged_records_rejected=damaged,
        result=pin(worker/'result.json'),weights=pin(worker/'weights.json'),frozen=pin(root/'frozen.json'),births=receipt['births'])
    write(BASE/'audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
