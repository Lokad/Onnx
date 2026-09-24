"""Reconcile projection observations and controls with every original public call."""
from collections import Counter
import importlib.util
import json
from run import ROOT, BASE, APP, MANIFEST, pin, read, write, prepared
from checks import qualify
from review_build import resources


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def main():
    prepared(); assert not (BASE/'closed.json').exists()
    resource_rows = resources('capture'); folder = BASE/'capture-collected'
    state = read(folder/'capture-state.json'); assert [r['name'] for r in state['runs']] == ['control','phase']
    spec = read(BASE/'bundle/spec.json'); built = read(BASE/'build-collected/built.json')
    review = read(BASE/'build-review.json')
    assert review['passed'] and review['core_unchanged'] and review['built'] == pin(BASE/'build-collected/built.json')
    assert review['spec'] == pin(BASE/'bundle/spec.json')
    manifest = read(APP/'collected'/MANIFEST)
    protocol = module('projection_app_protocol',APP/'collected/runtime/protocol.py')
    accounting = module('projection_accounting',APP/'collected/runtime/campaign_processes.py')
    previous = read(APP/'collected/timing-01-candidate/output/result.json')
    graph = read(BASE/'bundle/evidence/graphs.json')['encoder-model.onnx']['nodes']
    observations = []; geometry = Counter(); census = Counter(); times = {}; raw_clocks = []
    for run in state['runs']:
        mode = run['name']; value = read(folder/mode/'result.json')
        protocol.validate_records(value,manifest,'timing')
        assert run['accounting'] == accounting.foreign_fraction(run['cpu_before'],run['cpu_after'],state['supervisor']['pid'])
        assert run['accounting']['valid'] and run['accounting']['foreign_cpu_fraction'] <= .01
        assert value['passed'] and not value['sampled'] and value['runtime'] == '.NET 10.0.8'
        assert value['processor_count'] == 1 and value['affinity'] == 4 and not value['flags'] and value['held_outputs_unchanged']
        assert value['core_sha256'] == built['core']['sha256'] == spec['core']['sha256']
        assert value['data_sha256'] == built['data']['sha256'] and value['runner_sha256'] == built['consumer']['sha256']
        assert value['manifest_sha256'] == pin(APP/'collected'/MANIFEST)['sha256']
        assert len(value['records']) == len(list((folder/mode).glob('projection-*.json'))) == 80
        times[mode] = sum(r['seconds'] for r in value['records'] if r['phase']=='measured')/3
        for index,request in enumerate(value['records']):
            assert request == read(folder/mode/f'{index:03}.json')
            old = previous['records'][index]
            assert (request['name'],request['pass'],request['result']) == (old['name'],old['pass'],old['result'])
            assert request['thread_id'] == run['ready']['thread_id']
            raw_clocks.append(dict(mode=mode,index=index,**request))
            record = read(folder/mode/f'projection-{index:03}.json')
            assert (record['Name'],record['Pass'],record['Control']) == (request['name'],request['pass'],mode=='control')
            if mode == 'control':
                assert record['Calls'] == []; continue
            call, = record['Calls']
            assert 0 <= call['RetainedBytes'] <= call['MaximumBytes'] == 268435456
            rows = qualify(call,graph,record['Frequency'])
            census[(call['PackedWeights'],call['RetainedBytes'])] += 1
            frames = request['result']['encoded_frames']; geometry[frames] += len(rows)
            observations.extend(dict(request=index,clip=request['name'],pass_index=request['pass'],phase=request['phase'],frames=frames,**r) for r in rows)
    expected = Counter()
    for case in manifest['cases']: expected[case['expected']['encoded_frames']] += 4*217
    assert geometry == expected and len(geometry) == 19 and len(observations) == 17360
    analysis = dict(passed=True,requests=160,observations=17360,core_unchanged=True,exact_public_results=True,
        diagnostic_only=True,no_performance_score=True,actual_kernel_leaf_observed=False,
        measured_scratch_and_copy_requests=True,frame_counts=dict(sorted(geometry.items())),
        retained_census=[dict(weights=k[0],bytes=k[1],calls=v) for k,v in census.items()],
        corpus_seconds=times,observation_over_control=times['phase']/times['control'],overhead_subtracted=False,
        resources=resource_rows)
    write(BASE/'observations.json',observations); write(BASE/'clocks.json',raw_clocks); write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),observations=pin(BASE/'observations.json'),
        clocks=pin(BASE/'clocks.json'),build_review=pin(BASE/'build-review.json'),collection=pin(folder/'capture-collection.json'),
        transfer=pin(BASE/'capture-transfer.json'),auditor=pin(__file__),checks=pin(ROOT/'tests/parakeet/projection-route-amd/checks.py'),
        terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for run in state['runs'] for p,b in run['members'].items()]))
    print(json.dumps(analysis))


if __name__ == '__main__': main()
