"""Reconcile every original public request and every managed phase/node interval."""
from collections import Counter, defaultdict
import importlib.util
import json
import math
from run import ROOT, BASE, APP, pin, read, write, prepared

PHASES={'nemo128.onnx':'frontend','encoder-model.onnx':'encoder','decoder_joint-model.onnx':'decoder'}


def attribute(result, folder, mode):
    metadata=read(folder/'graphs.json')
    assert set(metadata)==set(PHASES)
    nodes={graph:{n['id']:n for n in record['nodes']} for graph,record in metadata.items()}
    assert all(len(nodes[g])==len(metadata[g]['nodes']) for g in metadata)
    totals=Counter();calls=Counter();node_ticks=Counter();node_calls=Counter();numeric_ops={};requests=[]
    measured_ticks=0;frequency=None
    for index,row in enumerate(result['records']):
        value=read(folder/f'phase-{index:03}.json')
        assert (value['name'],value['pass'],value['frequency'],value['mode'])==(row['name'],row['pass'],row['frequency'],mode)
        assert frequency in [None,row['frequency']];frequency=row['frequency']
        measured=row['phase']=='measured';phase_ticks=Counter();count=Counter();inside=Counter()
        previous=row['start_ticks']
        expected=['nemo128.onnx','encoder-model.onnx']+['decoder_joint-model.onnx']*row['result']['decoder_calls']
        assert [c['graph'] for c in value['calls']]==expected
        for call in value['calls']:
            graph=call['graph'];phase=PHASES[graph]
            a,b=call['start_ticks'],call['end_ticks']
            assert previous<=a<b<=row['end_ticks'];previous=b
            phase_ticks[phase]+=b-a;count[phase]+=1
            if mode=='phase':assert call['nodes'] is None;continue
            assert [n['NodeId'] for n in call['nodes']]==[n['id'] for n in metadata[graph]['nodes']]
            last=a
            for node in call['nodes']:
                key=(graph,node['NodeId']);start,end=node['StartTicks'],node['EndTicks']
                assert last<=start<=end<=b;last=end
                assert numeric_ops.setdefault(key,node['Op'])==node['Op']
                inside[phase]+=end-start
                if measured:node_ticks[key]+=end-start;node_calls[key]+=1
        duration=row['end_ticks']-row['start_ticks'];remainder=duration-sum(phase_ticks.values());assert remainder>=0
        if measured:
            measured_ticks+=duration;totals.update(phase_ticks);calls.update(count)
        requests.append(dict(name=row['name'],iteration=row['pass'],phase=row['phase'],ticks=duration,
            phase_ticks=dict(phase_ticks),call_counts=dict(count),node_ticks=dict(inside),remainder_ticks=remainder))
    assert len(requests)==80 and calls['frontend']==calls['encoder']==60 and calls['decoder']==3600
    node_rows=[]
    for (graph,id),ticks in node_ticks.items():
        node=nodes[graph][id]
        node_rows.append(dict(graph=PHASES[graph],**node,numeric_op=numeric_ops[(graph,id)],
            ticks=ticks,calls=node_calls[(graph,id)],corpus_seconds=ticks/(3*frequency)))
    return dict(corpus_seconds=measured_ticks/(3*frequency),
        phase_seconds={k:v/(3*frequency) for k,v in totals.items()},
        remainder_seconds=(measured_ticks-sum(totals.values()))/(3*frequency),
        call_counts=dict(calls),frequency=frequency,requests=requests,node_rows=node_rows)


def main():
    prepared()
    folder=BASE/'capture-collected';assert not (BASE/'closed.json').exists()
    transfer=read(BASE/'capture-transfer.json');receipt=read(folder/'capture-collection.json')
    assert transfer['passed'] and transfer['collection']==pin(folder/'capture-collection.json') and transfer['archive']==pin(BASE/'capture-results.tar.gz')
    assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/'capture-state.json');spec=read(BASE/'bundle/spec.json');built=read(folder/'built.json')
    assert state['supervisor']==read(BASE/'capture-deployment.json')
    review=read(BASE/'build-review.json')
    assert review['core_unchanged'] and review['consumer_unchanged'] and review['constructor_unchanged']
    assert built['core']==spec['core'] and built['consumer']==spec['original_consumer']
    reference=read(BASE/'bundle/evidence/candidate-public.json')
    expected={r['name']:r['result'] for r in reference['records']};assert len(expected)==20
    assert reference['core_sha256']==spec['core']['sha256'] and reference['data_sha256']==spec['data']['sha256']
    assert state['complete'] and state['code']==0 and [r['name'] for r in state['runs']]==['control','phase','wall']
    assert read(BASE/'build-review.json')['passed'] and read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    module=importlib.util.spec_from_file_location('original_protocol',APP/'collected/runtime/protocol.py')
    protocol=importlib.util.module_from_spec(module);module.loader.exec_module(protocol)
    module=importlib.util.spec_from_file_location('cpu_accounting',APP/'collected/runtime/campaign_processes.py')
    accounting=importlib.util.module_from_spec(module);module.loader.exec_module(accounting)
    manifest=read(APP/'collected/manifests/current-parakeet.json')
    resources=[];results={};phases={}
    for run in state['runs']:
        mode=run['name'];limits=spec['capture_limits']
        assert run['complete'] and run['code']==0 and run['seconds']<limits['seconds']
        assert run['preflight']['available']>=limits['available_before'] and run['preflight']['tmpfs']>=limits['tmpfs_before']
        samples=[json.loads(line) for line in (folder/'logs'/(mode+'.resources.jsonl')).read_text().splitlines()]
        assert samples and len(samples)==run['samples']
        for row in samples:
            assert row['seconds']<limits['seconds'] and row['rss']<limits['rss']
            assert row['available']>=spec['minimum_free'] and row['tmpfs']>=spec['minimum_free'] and row['output']<spec['output_limit']
            assert row['rss']==sum(m['rss'] for m in row['members'])
            for m in row['members']:
                assert run['members'][str(m['pid'])]==m['birth'] and m['affinity']==[2] and all(t==[2] for t in m['threads'])
        assert all(0<=b['seconds']-a['seconds']<10 for a,b in zip(samples,samples[1:]))
        assert run['accounting']==accounting.foreign_fraction(run['cpu_before'],run['cpu_after'],state['supervisor']['pid'])
        assert run['accounting']['valid'] and run['accounting']['foreign_cpu_fraction']<=.01
        value=read(folder/mode/'result.json');protocol.validate_records(value,manifest,'timing')
        assert value['passed'] and not value['sampled'] and value['runtime']=='.NET 10.0.8' and value['processor_count']==1
        assert value['core_sha256']==spec['core']['sha256'] and value['runner_sha256']==built['consumer']['sha256']
        assert value['data_sha256']==(spec['data'] if mode=='control' else built['data'])['sha256']
        assert value['manifest_sha256']==pin(APP/'collected/manifests/current-parakeet.json')['sha256']
        for index,row in enumerate(value['records']):
            assert row==read(folder/mode/f'{index:03}.json')
            assert row['thread_id']==run['ready']['thread_id']
            assert row['result']==expected[row['name']]
            if mode!='control':assert row['result']==results['control']['records'][index]['result']
        results[mode]=value
        if mode=='control':assert not list((folder/mode).glob('phase-*.json'))
        else:phases[mode]=attribute(value,folder/mode,mode)
        resources.append(dict(mode=mode,samples=len(samples),peak_rss=max(r['rss'] for r in samples),seconds=run['seconds']))
    corpus={mode:sum(r['seconds'] for r in value['records'] if r['phase']=='measured')/3 for mode,value in results.items()}
    assert read(folder/'phase/graphs.json')==read(folder/'wall/graphs.json')
    for mode,phase in phases.items():assert math.isclose(phase['corpus_seconds'],corpus[mode],rel_tol=1e-14)
    operators=defaultdict(float)
    for row in phases['wall']['node_rows']:operators[(row['graph'],row['op'])]+=row['corpus_seconds']
    analysis=dict(passed=True,corpus=corpus,phase_over_control=corpus['phase']/corpus['control'],
        wall_over_phase=corpus['wall']/corpus['phase'],phases=phases,resources=resources,
        operators=[dict(graph=g,op=op,corpus_seconds=seconds) for (g,op),seconds in sorted(operators.items(),key=lambda p:-p[1])],
        original_request_checks=True,core_unchanged=True,consumer_unchanged=True,constructor_unchanged=True,
        complete_admitted_public_results_exact=True,source_receipt=spec['source_receipt'],
        qualification_closures=spec['qualification_closures'],diagnostic_context=spec['diagnostic_context'],
        attribution_only=True,actual_kernel_dispatch_measured=False)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),build_review=pin(BASE/'build-review.json'),
        transfer=pin(BASE/'capture-transfer.json'),collection=pin(folder/'capture-collection.json'),auditor=pin(__file__),
        terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(corpus=corpus,phase_over_control=analysis['phase_over_control'],wall_over_phase=analysis['wall_over_phase'],
        phases={k:{n:v[n] for n in ['phase_seconds','remainder_seconds','call_counts']} for k,v in phases.items()},operators=analysis['operators'][:16])))


if __name__=='__main__':main()
