"""Recompute native coverage, array conformance and every retained resource check."""
import json
import numpy as np
from run import BASE, ROOT, prepared
from protocol import LIMITS, MODELS, check_sample, compare, graph, pin, profile_events, read, save


def main():
    proof = prepared(); assert not (BASE/'closed.json').exists()
    payload = BASE/'payload'; spec = read(payload/'payload.json'); collected = BASE/'collected'
    transfer = read(BASE/'collection-transfer.json'); receipt = read(collected/'collection.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for name,wanted in receipt['files'].items(): assert pin(collected/name) == wanted,name
    assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()} == set(receipt['files']) | {'collection.json'}
    state = read(collected/'identity.json'); assert state['complete'] and state['code'] == 0
    assert state['supervisor'] == read(BASE/'deployment.json') and state['boot_time'] == spec['boot_time']
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    assert [r['name'] for r in state['runs']] == ['metadata','profile']
    resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and 0 < row['seconds'] < LIMITS['seconds']
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        waits = read(collected/(row['name']+'-preflight.json'))
        assert waits == row['preflight_observations'] and row['preflight'] == waits[-1]
        assert all(0 <= r['seconds'] < 900 and r['tmpfs'] >= LIMITS['preflight_tmpfs'] for r in waits)
        samples = [json.loads(line) for line in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0
        assert max(r['rss'] for r in samples) == row['peak_rss'] and samples[-1]['seconds'] <= row['seconds']
        for sample in samples:
            check_sample(sample)
            for m in sample['members']: assert row['members'][str(m['pid'])] == m['birth']
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    metadata = read(collected/'metadata/result.json'); profile = read(collected/'profile/result.json')
    for value,mode,row in zip([metadata,profile],['metadata','profile'],state['runs'],strict=True):
        assert value['passed'] and value['mode'] == mode and value['inputs_and_held_outputs_unchanged']
        assert value['pid'] == row['child']['pid'] and value['birth'] == row['child']['birth'] and value['affinity'] == [2]
        assert value['interpreter'] == spec['interpreter'] and value['versions']['onnxruntime'] == '1.29.0'
        assert value['settings'] == dict(provider='CPUExecutionProvider',intra_threads=1,inter_threads=1,
            sequential=True,graph_optimizations='all',spinning=False,profiling=mode=='profile')
        assert set(value['models']) == set(MODELS)
        for origin in value['package_origins'].values(): assert origin in spec['external']
    assert metadata['versions'] == profile['versions'] and metadata['package_origins'] == profile['package_origins']
    assert metadata['arrays'] == 0 and profile['arrays'] == 12
    comparisons = []; models = {}
    local_models = dict(embedding=ROOT/'models/pyannote-embedding/embedding_encoder.onnx',
                        segmentation=ROOT/'models/pyannote-segmentation/segmentation/model.onnx')
    for model in MODELS:
        before = metadata['models'][model]; after = profile['models'][model]
        assert before['original'] == after['original'] and before['inputs'] == after['inputs'] and not before['records']
        assert before['original'] == pin(local_models[model]) == spec['external'][spec['models'][model]]
        censuses = {}
        for mode,result in [('metadata',before),('profile',after)]:
            target = collected/mode/(model+'-optimized.onnx')
            assert pin(target) == result['optimized']; censuses[mode] = graph(target)
        assert before['optimized'] == after['optimized'], 'Profiling altered the optimized graph'
        trace = collected/'profile'/after['profile']; assert pin(trace) == after['profile_pin']
        events = profile_events(read(trace),censuses['profile']); assert events['events'] == after['events']
        save(BASE/(model+'-graph-and-execution.json'),dict(original=graph(local_models[model]),optimized=censuses['profile'],execution=events))
        cases = [c for c in spec['cases'] if c['model'] == model]
        assert [(r['name'],r['model'],r['repeat']) for r in after['records']] == [(c['name'],model,i) for i in range(2) for c in cases]
        held = {}
        for row in after['records']:
            case = next(c for c in cases if c['name'] == row['name'])
            assert row['input_pin'] == case['input_pin'] == pin(payload/case['input'])
            assert pin(payload/case['reference']) == case['reference_pin']
            path = collected/'profile'/row['output']; assert pin(path) == row['output_pin']
            actual = np.load(path,allow_pickle=False); expected = np.load(payload/case['reference'],allow_pickle=False)
            checked = compare(actual,expected); assert checked == row['comparison'] and checked['failed_values'] == 0
            if row['name'] in held: assert actual.tobytes() == held[row['name']]
            held[row['name']] = actual.tobytes()
            comparisons.append(dict(name=row['name'],model=model,repeat=row['repeat'],**checked))
        models[model] = dict(original_nodes=graph(local_models[model])['top_level_nodes'],optimized_nodes=censuses['profile']['top_level_nodes'],
            counts=censuses['profile']['counts'],executed_kernel_events=len(events['kernels']),runs=events['model_runs'])
    analysis = dict(passed=True,models=models,arrays=len(comparisons),values=sum(c['values'] for c in comparisons),
        maximum=max(c['maximum'] for c in comparisons),comparisons=comparisons,resources=resources,
        resource_samples=sum(r['samples'] for r in resources),peak_rss=max(r['peak_rss'] for r in resources),
        scope='AMD optimized graph and executed operator evidence. Profile durations are diagnostic; no application speedup or assembly microkernel claim.')
    save(BASE/'analysis.json',analysis)
    files = {p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,files=files,local_inputs=proof['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),models=models,arrays=analysis['arrays'],maximum=analysis['maximum'],resources=analysis['resource_samples'])))


if __name__ == '__main__': main()
