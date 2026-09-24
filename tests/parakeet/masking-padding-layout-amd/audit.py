"""Reconcile every actual layout and mask with the original complete requests."""
from collections import Counter
import importlib.util
import json
from run import ROOT, BASE, APP, MANIFEST, pin, read, write, prepared
from checks import KINDS, qualify
from review_build import resources


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def main():
    prepared()
    assert not (BASE/'closed.json').exists()
    resource_rows = resources('capture'); folder = BASE/'capture-collected'
    state = read(folder/'capture-state.json'); run, = state['runs']; assert run['name'] == 'phase'
    spec = read(BASE/'bundle/spec.json'); built = read(BASE/'build-collected/built.json')
    review = read(BASE/'build-review.json')
    assert review['passed'] and review['built'] == pin(BASE/'build-collected/built.json')
    assert review['spec'] == pin(BASE/'bundle/spec.json') and review['core_unchanged']
    manifest = read(APP/'collected'/MANIFEST)
    protocol = module('application_protocol',APP/'collected/runtime/protocol.py')
    accounting = module('application_accounting',APP/'collected/runtime/campaign_processes.py')
    assert run['accounting'] == accounting.foreign_fraction(run['cpu_before'],run['cpu_after'],state['supervisor']['pid'])
    assert run['accounting']['valid'] and run['accounting']['foreign_cpu_fraction'] <= .01
    result = read(folder/'phase/result.json'); protocol.validate_records(result,manifest,'timing')
    assert result['passed'] and not result['sampled'] and result['runtime'] == '.NET 10.0.8' and result['processor_count'] == 1
    assert result['affinity'] == 4 and not result['flags'] and result['held_outputs_unchanged']
    assert result['core_sha256'] == built['core']['sha256'] == spec['core']['sha256']
    assert result['data_sha256'] == built['data']['sha256'] and result['runner_sha256'] == built['consumer']['sha256']
    assert result['manifest_sha256'] == pin(APP/'collected'/MANIFEST)['sha256']
    previous = read(APP/'collected/timing-01-candidate/result.json')
    graph = read(BASE/'bundle/evidence/graphs.json')['encoder-model.onnx']['nodes']
    assert len(graph) == 2856
    assert len(list((folder/'phase').glob('layout-*.json'))) == len(result['records']) == 80
    observations = []; geometry = Counter(); mask_counts = Counter(); layouts = Counter()
    for index, request in enumerate(result['records']):
        assert request == read(folder/'phase'/f'{index:03}.json')
        old = previous['records'][index]
        assert (request['name'],request['pass'],request['result']) == (old['name'],old['pass'],old['result'])
        assert request['thread_id'] == run['ready']['thread_id']
        record = read(folder/'phase'/f'layout-{index:03}.json')
        assert (record['Name'],record['Pass']) == (request['name'],request['pass'])
        call, = record['Calls']; assert call['GraphNodes'] == 2856
        frames = request['result']['encoded_frames']
        rows = qualify(call['Records'],frames,graph)
        for row in rows:
            geometry[frames] += 1
            if row['mask'] is not None: mask_counts[(row['family'],row['mask'])] += 1
            for operand, description in [*enumerate(row['inputs']),('output',row['output'])]:
                if description is not None:
                    layouts[(row['family'],str(operand),json.dumps(description,sort_keys=True))] += 1
            observations.append(dict(request=index,**row))
    expected = Counter()
    for case in manifest['cases']: expected[case['expected']['encoded_frames']] += 4*120
    assert geometry == expected and len(geometry) == 19
    assert len(observations) == 9600 and Counter(r['family'] for r in observations) == {kind:1920 for kind,_ in KINDS}
    analysis = dict(passed=True,requests=80,observations=9600,core_unchanged=True,exact_public_results=True,
        diagnostic_only=True,no_performance_score=True,frame_observation_counts=dict(sorted(geometry.items())),
        masks=[dict(family=k[0],mask=k[1],observations=v) for k,v in sorted(mask_counts.items())],
        layouts=[dict(family=k[0],operand=k[1],layout=json.loads(k[2]),observations=v) for k,v in sorted(layouts.items())],
        resources=resource_rows)
    write(BASE/'observations.json',observations)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),observations=pin(BASE/'observations.json'),
        build_review=pin(BASE/'build-review.json'),collection=pin(folder/'capture-collection.json'),
        transfer=pin(BASE/'capture-transfer.json'),auditor=pin(__file__),checks=pin(ROOT/'tests/parakeet/masking-padding-layout-amd/checks.py'),
        terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for p,b in run['members'].items()]))
    print(json.dumps(analysis))


if __name__ == '__main__': main()
