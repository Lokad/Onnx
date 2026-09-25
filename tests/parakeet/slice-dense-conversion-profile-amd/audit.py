"""Reconcile complete profiles and test the prospective positional-copy prediction."""
import importlib.util
import json
import math
from run import ROOT,BASE,TOOLS,APP,GRAPH_REFERENCE,MANIFEST,pin,read,write,prepared,observer_scope
from groups import compare_positional


def module(name,path):
    loader=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(loader);loader.loader.exec_module(result);return result


def main():
    prepared();assert not (BASE/'closed.json').exists()
    folder=BASE/'capture-collected';spec=read(BASE/'bundle/spec.json')
    transfer=read(BASE/'capture-transfer.json');receipt=read(folder/'capture-collection.json')
    assert transfer['passed'] and transfer['collection']==pin(folder/'capture-collection.json')
    assert transfer['archive']==pin(BASE/'capture-results.tar.gz') and receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/'capture-state.json');assert state['complete'] and state['code']==0
    assert state['supervisor']==read(BASE/'capture-deployment.json') and receipt['state']==pin(folder/'capture-state.json')
    assert [r['name'] for r in state['runs']]==['control','wall']
    assert pin(folder/'spec.json')==pin(BASE/'bundle/spec.json')
    runtime=read(folder/'runtime.json');review=read(folder/'observer-review.json')
    assert pin(folder/'runtime.json')==pin(BASE/'bundle/runtime.json')==review['runtime']
    assert pin(folder/'observer-review.json')==pin(BASE/'bundle/observer-review.json')
    scope=observer_scope();assert all(review[k]==v for k,v in scope.items())
    protocol=module('original_protocol',APP/'collected/runtime/protocol.py')
    accounting=module('cpu_accounting',APP/'collected/runtime/campaign_processes.py')
    attribution=module('original_attribution',TOOLS.parent/'managed-phase-amd/audit.py')
    manifest=read(APP/'collected'/MANIFEST);results={};phases={};resources=[]
    for run in state['runs']:
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
    assert read(folder/'control/graphs.json')==read(folder/'wall/graphs.json')==read(GRAPH_REFERENCE)==read(BASE/'bundle/expected-graphs.json')
    model=ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    matched=read(BASE/'bundle/positional-groups.json');assert pin(BASE/'bundle/positional-groups.json')==spec['groups']
    assert pin(model)==matched['source_model']=={k:manifest['models']['encoder-model.onnx'][k] for k in ['bytes','sha256']}
    comparison=compare_positional(phases['selected'],phases['candidate'],matched,model)
    frames={r['expected']['encoded_frames'] for r in manifest['cases']};assert len(frames)==19 and len(manifest['cases'])==20
    assert spec['require_complete_group_improves'] and spec['require_all_24_kernels_improve']
    passed=comparison['passed']
    analysis=dict(passed=passed,original_request_checks=True,requests=160,clips=20,frames=sorted(frames),
        phases=phases,positional=comparison,resources=resources,
        prospective_groups=spec['groups'],prediction='Complete positional group and all24 MatMul kernels improve',
        observer_review=pin(folder/'observer-review.json'),observer_reused_exactly=True,no_rebuild=True,
        attribution_only=True,application_gain_admitted=False,overhead_subtracted=False,independent_candidate_overhead_measured=False)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=passed,analysis=pin(BASE/'analysis.json'),observer_review=pin(folder/'observer-review.json'),
        transfer=pin(BASE/'capture-transfer.json'),collection=pin(folder/'capture-collection.json'),auditor=pin(__file__),
        terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]))
    print(json.dumps(dict(passed=passed,closed=pin(BASE/'closed.json'),
        positional={k:v for k,v in comparison.items() if k not in ['rows','all_nodes']},resources=resources)))
    assert passed,'Positional-copy prediction failed; retain all results and inspect before any scored comparison'


if __name__=='__main__':main()
