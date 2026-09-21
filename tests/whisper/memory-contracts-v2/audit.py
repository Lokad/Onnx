"""Audit complete public request transitions, pinned references and resource ownership."""
from pathlib import Path
import copy,importlib.util,json,math,sys
from prepare import ROOT,BASE,PRIOR,PRODUCT,pin,read,write
sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil
from tokenizers import Tokenizer
sys.path.insert(0,str(ROOT/'tests/whisper/weight-sharing-v2'))
from sharing_protocol import validate_sharing

spec=importlib.util.spec_from_file_location('original_recording_policy_audit',ROOT/'tests/whisper/recording/audit.py')
recording=importlib.util.module_from_spec(spec);spec.loader.exec_module(recording)


def terminal(births):
    for b in births:
        try:assert psutil.Process(b['pid']).create_time()!=b['birth']
        except psutil.NoSuchProcess:pass


def short_decisions(actual,expected):
    assert actual['text']==expected['text'] and actual['token_ids']==expected['tokens']
    assert actual['stop_reason']==expected['stop_reason'] and actual['skipped_as_no_speech']==expected['skipped_as_no_speech']


def validate_application(value,inputs,native,old,short,tokenizer):
    assert value['schema']==1 and value['refusals']==16 and value['ownership'] is True and not value['flags']
    assert [r['name'] for r in value['cases']]==['connected','shifted','token-limit','window-limit','connected']
    assert [r['repeat'] for r in value['cases']]==[False]*4+[True]
    assert value['cases'][0]['result']==value['cases'][4]['result']
    observations=[]
    for index,row in enumerate(value['cases']):
        case=inputs['cases'][index%4];reference=native['cases'][index%4]
        assert row['ownership'] is True and row['pcm_sha256']==case['pcm_sha256']==reference['pcm_sha256']
        assert math.isfinite(row['seconds']) and row['seconds']>0
        recording.validate_recording(row['result'],case,tokenizer)
        assert recording.decisions(row['result'])==recording.decisions(reference['result'])
        confidence=[]
        for a,b in zip(row['result']['windows'],reference['result']['windows'],strict=True):
            for key in ['no_speech_probability','average_log_probability']:
                if a['decoding'][key] is not None and b['decoding'][key] is not None:
                    confidence.append(abs(a['decoding'][key]-b['decoding'][key]))
        observations.append(dict(name=row['name'],repeat=row['repeat'],windows=len(row['result']['windows']),segments=len(row['result']['segments']),
            stop_reason=row['result']['stop_reason'],native_decisions_equal=True,max_observed_confidence_difference=max(confidence,default=0)))
    for key in ['empty','silent','concurrent']:assert value[key]==old[key],key
    short_decisions(value['short_regression'],short['cases'][0])
    assert value['short_recovery']==value['short_regression']
    concurrent=value['concurrent_speech'];assert len(concurrent)==2 and [c['index'] for c in concurrent]==[0,1]
    assert type(value['frequency']) is int and value['frequency']>0
    for row,case in zip(concurrent,short['cases'][:2],strict=True):
        assert row['pcm_sha256']==case['pcm_sha256']
        assert type(row['start']) is int and type(row['end']) is int and 0<row['start']<row['end']
        short_decisions(row['result'],case)
    assert max(c['start'] for c in concurrent)<min(c['end'] for c in concurrent)
    assert concurrent[0]['result']==value['short_regression']
    calls=len(value['cases'])+len(value['concurrent'])+len(concurrent)+4;assert calls==13
    return dict(calls=calls,refusals=16,recordings=observations,concurrent_speech_calls=2,
        overlap_seconds=(min(c['end'] for c in concurrent)-max(c['start'] for c in concurrent))/value['frequency'],
        short_recovery_exact=True,concurrent_first_exact=True)


def refuses(action):
    try:action()
    except (AssertionError,ValueError,KeyError,TypeError,IndexError):return
    raise AssertionError('Damaged evidence accepted')


def main():
    folder=BASE/'local';state=read(folder/'identity.json');frozen=read(folder/'frozen.json');prepared=read(BASE/'prepared.json')
    assert state['complete'] and state['code']==0 and 'error' not in state and state['frozen']==pin(folder/'frozen.json')
    births=[state['supervisor']]+[dict(pid=int(pid),birth=birth) for pid,birth in state['members'].items()];terminal(births)
    assert frozen['prepared']==pin(BASE/'prepared.json') and frozen['supervisor']==pin(folder/'run_local.py')
    for name,wanted in prepared['files'].items():assert pin(BASE/name)==wanted,name
    for name,wanted in prepared['inputs'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in frozen['external'].items():assert pin(Path(name))==wanted,name
    worker=folder/'worker';value=read(worker/'result.json');assert pin(worker/'result.json')==state['result']
    assert value['runtime']=='.NET 10.0.12' and value['affinity']==4 and value['processor_count']==1
    for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','WhisperMemoryContractsV2.dll')]:assert value[key]==pin(BASE/'bin'/name)['sha256']
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:
        assert pin(BASE/'bin'/name)==pin(PRODUCT/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'/name)
    inputs=read(PRIOR/'inputs/inputs.json');native=read(PRIOR/'native-corrected/manifest.json');old=read(PRIOR/'managed/result.json')
    short_path=ROOT/'artifacts/asr-labeled-20260919/native-whisper/manifest.json';short=read(short_path)
    assert value['inputs_sha256']==native['inputs_sha256']==pin(PRIOR/'inputs/inputs.json')['sha256']
    assert value['short_manifest_sha256']==pin(short_path)['sha256']
    tokenizer=Tokenizer.from_file(str(ROOT/'models/whisper-large-v3-turbo/tokenizer.json'))
    application=validate_application(value,inputs,native,old,short,tokenizer)
    for index,row in enumerate(value['cases']):assert read(worker/f"{index:02}-{row['name']}.json")==row
    weights=read(worker/'weights.json');assert pin(worker/'weights.json')['sha256']==value['weights_sha256']
    for stage in ['before','after']:assert read(worker/('weights-'+stage+'.json'))==weights[stage]
    sharing=validate_sharing(dict(weight_sharing=weights));census=read(PRODUCT/'weight-census.json');checked=0
    for stage in ['before','after']:
        for name,original in zip(['first','past'],census['graphs'],strict=True):
            entries={r['name']:r for r in weights[stage][name]['initializers']}
            for row in original['rows']:
                actual=entries[row['name']]
                assert actual['bytes']==row['bytes'] and actual['shape']==row['shape'] and actual['sha256']==row['sha256']
                assert actual['type']=={1:'Float',7:'Int64'}[row['type']];checked+=1
    expected={f"{i:02}-{r['name']}.json" for i,r in enumerate(value['cases'])}|{'result.json','weights.json','weights-before.json','weights-after.json'}
    assert {p.name for p in worker.iterdir()}==expected and checked==1136
    samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()];limits=frozen['limits']
    assert state['limits']==limits and len(samples)==state['samples'] and 0<state['seconds']<limits['seconds']
    assert state['preflight_available']>=limits['preflight'] and state['preflight_disk']>=limits['preflight_disk']
    preflight=read(folder/'preflight.json');assert preflight['available']==state['preflight_available'] and preflight['disk']==state['preflight_disk']
    previous=0;seen={};gaps=[]
    for sample in samples:
        assert previous<sample['seconds']<state['seconds'];gaps.append(sample['seconds']-previous);previous=sample['seconds']
        assert sample['available']>=limits['available'] and sample['disk']>=limits['disk']
        assert sample['members'] and sum(m['rss'] for m in sample['members'])<limits['rss']
        for member in sample['members']:
            assert member['affinity']==[2] and member['rss']>=0 and member['birth']==state['members'][str(member['pid'])]
            seen[str(member['pid'])]=member['birth']
    gaps.append(state['seconds']-previous);assert max(gaps)<10 and seen==state['members']
    peak=max(sum(m['rss'] for m in s['members']) for s in samples);assert peak==state['peak_rss']
    damaged=0
    for edit in [lambda v:v.update(refusals=15),lambda v:v['cases'].pop(),lambda v:v['cases'][0].update(ownership=False),
        lambda v:v['cases'][0]['result']['segments'][0].update(text='wrong'),lambda v:v['short_recovery'].update(text='wrong'),
        lambda v:v['concurrent_speech'][1].update(start=v['concurrent_speech'][0]['end']+1,end=v['concurrent_speech'][0]['end']+2),
        lambda v:v['concurrent_speech'][1].update(pcm_sha256='0'*64),lambda v:v['concurrent_speech'][1]['result']['token_ids'].pop(),
        lambda v:v['silent'].update(processed_seconds=599)]:
        bad=copy.deepcopy(value);edit(bad);refuses(lambda:validate_application(bad,inputs,native,old,short,tokenizer));damaged+=1
    for edit in [lambda v:v['after']['first']['initializers'][0].update(sha256='0'*64),lambda v:v['after'].update(unique_arrays=1),
        lambda v:next(r for r in v['after']['past']['initializers'] if r['name']=='folded:Transpose_801').update(tensor_name='wrong')]:
        bad=copy.deepcopy(weights);edit(bad);refuses(lambda:validate_sharing(dict(weight_sharing=bad)));damaged+=1
    result=dict(passed=True,private_prototype=True,benchmark=False,application=application,sharing=sharing,original_initializer_checks=checked,
        resources=dict(samples=len(samples),peak_rss=peak,min_available=min(s['available'] for s in samples),min_disk=min(s['disk'] for s in samples),max_gap=max(gaps)),
        damaged_records_rejected=damaged,result=pin(worker/'result.json'),weights=pin(worker/'weights.json'),frozen=pin(folder/'frozen.json'),births=births)
    write(BASE/'local-audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
