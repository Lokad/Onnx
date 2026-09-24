"""Reconcile all requests/nodes and test the prospective complete-pair prediction."""
import importlib.util
import json
import math
from run import ROOT, BASE, TOOLS, APP, PRIOR, pin, read, write
from review_build import resources


def module(name,path):
    loader=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(loader);loader.loader.exec_module(result);return result


def compare_pairs(selected,candidate):
    rows=[]
    old={n['name']:n for n in selected['node_rows'] if n['graph']=='encoder'}
    new={n['name']:n for n in candidate['node_rows'] if n['graph']=='encoder'}
    assert old.keys()==new.keys()
    for name,row in old.items():
        assert {k:v for k,v in row.items() if k not in ['ticks','corpus_seconds']}=={
            k:v for k,v in new[name].items() if k not in ['ticks','corpus_seconds']},name
    for layer in range(24):
        prefix=f'/layers.{layer}/self_attn/'
        names=[prefix+'Slice_1',prefix+'Reshape_7']
        for name,op in zip(names,['Slice','Reshape']):
            assert old[name]['op']==new[name]['op']==op
            assert old[name]['calls']==new[name]['calls']==60
        assert old[names[0]]['outputs'][0]==old[names[1]]['inputs'][0]
        rows.append(dict(layer=layer,selected_seconds=sum(old[n]['corpus_seconds'] for n in names),
            candidate_seconds=sum(new[n]['corpus_seconds'] for n in names),
            nodes=[dict(name=n,selected_seconds=old[n]['corpus_seconds'],
                candidate_seconds=new[n]['corpus_seconds']) for n in names]))
    before=sum(r['selected_seconds'] for r in rows);after=sum(r['candidate_seconds'] for r in rows)
    assert before>0 and after>=0
    return dict(layers=24,nodes=48,selected_seconds=before,candidate_seconds=after,gain=1-after/before,rows=rows)


def main():
    assert not (BASE/'closed.json').exists()
    resource_rows=resources('capture');folder=BASE/'capture-collected'
    transfer=read(BASE/'capture-transfer.json');receipt=read(folder/'capture-collection.json')
    assert transfer['passed'] and transfer['collection']==pin(folder/'capture-collection.json')
    assert transfer['archive']==pin(BASE/'capture-results.tar.gz')
    state=read(folder/'capture-state.json');spec=read(BASE/'bundle/spec.json');built=read(folder/'built.json')
    assert [r['name'] for r in state['runs']]==['control','wall']
    assert read(BASE/'build-review.json')['passed'] and read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    protocol=module('original_protocol',APP/'collected/runtime/protocol.py')
    accounting=module('cpu_accounting',APP/'collected/runtime/campaign_processes.py')
    attribution=module('original_attribution',TOOLS.parent/'managed-phase-amd/audit.py')
    manifest=read(APP/'collected/manifests/current-parakeet.json')
    results={};phases={}
    for run in state['runs']:
        name=run['name'];role='selected' if name=='control' else 'candidate'
        assert run['accounting']==accounting.foreign_fraction(run['cpu_before'],run['cpu_after'],state['supervisor']['pid'])
        assert run['accounting']['valid'] and run['accounting']['foreign_cpu_fraction']<=.01
        value=read(folder/name/'result.json');protocol.validate_records(value,manifest,'timing')
        assert value['passed'] and not value['sampled'] and value['runtime']=='.NET 10.0.8' and value['processor_count']==1
        assert value['core_sha256']==spec['core' if role=='selected' else 'candidate_core']['sha256']
        assert value['runner_sha256']==built['consumer']['sha256'] and value['data_sha256']==spec['data']['sha256']
        assert value['manifest_sha256']==pin(APP/'collected/manifests/current-parakeet.json')['sha256']
        for index,row in enumerate(value['records']):
            assert row==read(folder/name/f'{index:03}.json') and row['thread_id']==run['ready']['thread_id']
            if role=='candidate':assert row['result']==results['selected']['records'][index]['result']
        results[role]=value;phases[role]=attribution.attribute(value,folder/name,'wall')
        assert math.isclose(phases[role]['corpus_seconds'],
            sum(r['seconds'] for r in value['records'] if r['phase']=='measured')/3,rel_tol=1e-14)
    assert read(folder/'control/graphs.json')==read(folder/'wall/graphs.json')
    # Bind these pairs to the already reconciled original and optimized ORT graphs.
    assert read(folder/'control/graphs.json')==read(PRIOR/'capture-collected/wall/graphs.json')
    pairs=compare_pairs(phases['selected'],phases['candidate'])
    passed=pairs['gain']>=spec['minimum_pair_gain']
    analysis=dict(passed=passed,original_request_checks=True,requests=160,phases=phases,pairs=pairs,
        threshold=spec['minimum_pair_gain'],resources=resource_rows,attribution_only=True,
        application_gain_admitted=False,overhead_subtracted=False,independent_candidate_overhead_measured=False)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=passed,analysis=pin(BASE/'analysis.json'),build_review=pin(BASE/'build-review.json'),
        transfer=pin(BASE/'capture-transfer.json'),collection=pin(folder/'capture-collection.json'),auditor=pin(__file__),
        terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]))
    print(json.dumps(dict(passed=passed,closed=pin(BASE/'closed.json'),pairs={k:v for k,v in pairs.items() if k!='rows'},
        corpus={k:v['corpus_seconds'] for k,v in phases.items()},resources=resource_rows)))
    assert passed,'Causal prediction failed; retain all results, no unchanged retry'


if __name__=='__main__':main()
