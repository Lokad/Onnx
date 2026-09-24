"""Reconcile complete profiles and test the prospective masking prediction."""
import importlib.util
import json
import math
from run import ROOT,BASE,TOOLS,APP,PRIOR,MATCHED,MANIFEST,pin,read,write,prepared,observer_scope,ORIGINAL,initial
from run import ORIGINAL_TOOLS


def module(name,path):
    loader=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(loader);loader.loader.exec_module(result);return result


compare_where=module('original_masking_groups',ORIGINAL_TOOLS/'groups.py').compare_where

def main():
    prepared();assert not (BASE/'closed.json').exists()
    folder=BASE/'capture-collected';spec=read(BASE/'bundle/spec.json')
    transfer=read(BASE/'capture-transfer.json');receipt=read(folder/'capture-collection.json')
    assert transfer['passed'] and transfer['collection']==pin(folder/'capture-collection.json')
    assert transfer['archive']==pin(BASE/'capture-results.tar.gz') and receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/'capture-state.json');assert state['complete'] and state['code']==0
    assert state['supervisor']==read(BASE/'capture-deployment.json') and receipt['state']==pin(folder/'capture-state.json')
    assert [r['name'] for r in state['runs']]==['wall']
    first=initial();first_folder=ORIGINAL/'capture-collected';first_state=read(first_folder/'capture-state.json')
    recovery_folder=folder;recovery_state=state
    assert first_state['ended']<recovery_state['started']
    assert pin(first_folder/'runtime.json')==pin(folder/'runtime.json')
    assert pin(first_folder/'observer-review.json')==pin(folder/'observer-review.json')
    assert pin(folder/'spec.json')==pin(BASE/'bundle/spec.json')
    runtime=read(folder/'runtime.json');review=read(folder/'observer-review.json')
    assert pin(folder/'runtime.json')==pin(BASE/'bundle/runtime.json')==review['runtime']
    assert pin(folder/'observer-review.json')==pin(BASE/'bundle/observer-review.json')
    scope=observer_scope();assert all(review[k]==v for k,v in scope.items())
    protocol=module('original_protocol',APP/'collected/runtime/protocol.py')
    accounting=module('cpu_accounting',APP/'collected/runtime/campaign_processes.py')
    attribution=module('original_attribution',TOOLS.parent/'managed-phase-amd/audit.py')
    manifest=read(APP/'collected'/MANIFEST);results={};phases={};resources=[]
    for folder,state,run in [(first_folder,first_state,first_state['runs'][0]),(recovery_folder,recovery_state,recovery_state['runs'][0])]:
        name=run['name'];role='selected' if name=='control' else 'candidate';limits=spec['capture_limits']
        assert run['complete'] and run['code']==0 and run['seconds']<limits['seconds']
        assert run['preflight']['available']>=limits['available_before'] and run['preflight']['tmpfs']>=limits['tmpfs_before']
        samples=[json.loads(s) for s in (folder/'logs'/(name+'.resources.jsonl')).read_text().splitlines()]
        assert samples and len(samples)==run['samples']
        for sample in samples:
            assert sample['seconds']<limits['seconds'] and sample['rss']<limits['rss']
            assert sample['available']>=spec['minimum_free'] and sample['tmpfs']>=spec['minimum_free'] and sample['output']<spec['output_limit']
            assert sample['rss']==sum(m['rss'] for m in sample['members'])
            for m in sample['members']:
                assert run['members'][str(m['pid'])]==m['birth'] and m['affinity']==[2] and all(t==[2] for t in m['threads'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[run['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        assert run['accounting']==accounting.foreign_fraction(run['cpu_before'],run['cpu_after'],state['supervisor']['pid'])
        assert run['accounting']['valid'] and run['accounting']['foreign_cpu_fraction']<=.01
        value=read(folder/name/'result.json');protocol.validate_records(value,manifest,'timing')
        assert value['passed'] and not value['sampled'] and value['runtime']=='.NET 10.0.8' and value['processor_count']==1
        assert value['core_sha256']==spec['core' if role=='selected' else 'candidate_core']['sha256']
        assert value['runner_sha256']==runtime['consumer']['sha256'] and value['data_sha256']==spec['data']['sha256']
        assert value['manifest_sha256']==pin(APP/'collected'/MANIFEST)['sha256']
        for index,row in enumerate(value['records']):
            assert row==read(folder/name/f'{index:03}.json') and row['thread_id']==run['ready']['thread_id']
            if role=='candidate':assert row['result']==results['selected']['records'][index]['result']
        results[role]=value;phases[role]=attribution.attribute(value,folder/name,'wall')
        assert math.isclose(phases[role]['corpus_seconds'],sum(r['seconds'] for r in value['records'] if r['phase']=='measured')/3,rel_tol=1e-14)
        resources.append(dict(name=name,samples=len(samples),peak_rss=max(r['rss'] for r in samples),seconds=run['seconds']))
    assert read(first_folder/'control/graphs.json')==read(folder/'wall/graphs.json')==read(PRIOR/'capture-collected/wall/graphs.json')
    model=ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    matched=read(BASE/'bundle/matched-input-groups.json');assert matched==read(MATCHED/'analysis.json')
    assert pin(model)==matched['original_model']=={k:manifest['models']['encoder-model.onnx'][k] for k in ['bytes','sha256']}
    comparison=compare_where(phases['selected'],phases['candidate'],matched,model)
    frames={r['expected']['encoded_frames'] for r in manifest['cases']};assert len(frames)==19 and len(manifest['cases'])==20
    passed=comparison['gain']>=spec['minimum_where_gain'] and all(r['complete_group_gain']>0 for r in comparison['families'])
    analysis=dict(passed=passed,original_request_checks=True,requests=160,clips=20,frames=sorted(frames),
        phases=phases,masking=comparison,threshold=spec['minimum_where_gain'],resources=resources,
        observer_review=pin(folder/'observer-review.json'),observer_reused_exactly=True,no_rebuild=True,
        split_capture=True,initial_failure=first,completed_control_repeated=False,
        attribution_only=True,application_gain_admitted=False,overhead_subtracted=False,independent_candidate_overhead_measured=False)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=passed,analysis=pin(BASE/'analysis.json'),observer_review=pin(folder/'observer-review.json'),
        transfer=pin(BASE/'capture-transfer.json'),collection=pin(folder/'capture-collection.json'),auditor=pin(__file__),initial=first,
        terminal_owners=[s['supervisor'] for s in [first_state,recovery_state]]+[dict(pid=int(p),birth=b) for s in [first_state,recovery_state] for r in s['runs'] for p,b in r['members'].items()]))
    print(json.dumps(dict(passed=passed,closed=pin(BASE/'closed.json'),
        masking={k:v for k,v in comparison.items() if k not in ['rows','union_rows']},resources=resources)))
    assert passed,'Masking prediction failed; retain all results and inspect before any scored comparison'


if __name__=='__main__':main()
