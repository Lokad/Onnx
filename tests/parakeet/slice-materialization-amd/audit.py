"""Reconcile all actual slice layouts against the original corpus and source hypothesis."""
from collections import Counter
import importlib.util
import json
from run import ROOT, BASE, APP, pin, read, write
from review_build import resources


def qualify(row, frames, layer):
    t = frames
    assert row['Name'] == f'/layers.{layer}/self_attn/Slice_1_output_0'
    assert row['ElementType'] == 'System.Single'
    assert row['Dimensions'] == [1,8,2*t-1,t] and row['Destination'] == [1,8,t,2*t-1]
    assert row['Strides'] == [8*(2*t-1)*t,(2*t-1)*t,t,1] and not row['Reversed']
    assert row['Dense'] and row['ParentType'].startswith('Lokad.Onnx.DenseTensor`1[[System.Single,')
    assert row['ParentName'] == f'/layers.{layer}/self_attn/Reshape_6_output_0'
    assert row['ParentDimensions'] == [1,8,2*t,t] and not row['ParentReversed']
    assert row['ParentStrides'] == [16*t*t,2*t*t,t,1]
    assert row['ParentLength'] == row['BufferLength'] == row['ArrayCount'] == 16*t*t
    assert row['ArrayBacked'] and 0 <= row['ArrayOffset'] <= row['ArrayLength']-row['ArrayCount']
    assert row['Definitions'] == [dict(Start=0,Step=1,Count=1,IsIndex=False),
        dict(Start=0,Step=1,Count=8,IsIndex=False),dict(Start=1,Step=1,Count=2*t-1,IsIndex=False),
        dict(Start=0,Step=1,Count=t,IsIndex=False)]
    length = (2*t-1)*t
    spans = [dict(source_offset=head*2*t*t+t,destination_offset=head*length,length=length) for head in range(8)]
    assert sum(s['length'] for s in spans) == 8*length
    assert all(0 <= s['source_offset'] <= row['BufferLength']-s['length'] for s in spans)
    assert all(a['source_offset']+a['length']+t == b['source_offset'] for a,b in zip(spans,spans[1:]))
    return spans


def main():
    resource_rows = resources('capture'); folder = BASE/'capture-collected'
    state = read(folder/'capture-state.json'); run, = state['runs']
    spec = read(BASE/'bundle/spec.json'); built = read(BASE/'build-collected/built.json')
    assert read(BASE/'build-review.json')['passed']
    manifest = read(APP/'collected/manifests/current-parakeet.json')
    loader = importlib.util.spec_from_file_location('application_protocol',APP/'collected/runtime/protocol.py')
    protocol = importlib.util.module_from_spec(loader); loader.loader.exec_module(protocol)
    loader = importlib.util.spec_from_file_location('application_accounting',APP/'collected/runtime/campaign_processes.py')
    accounting = importlib.util.module_from_spec(loader); loader.loader.exec_module(accounting)
    assert run['accounting'] == accounting.foreign_fraction(run['cpu_before'],run['cpu_after'],state['supervisor']['pid'])
    assert run['accounting']['valid'] and run['accounting']['foreign_cpu_fraction'] <= .01
    result = read(folder/'phase/result.json'); protocol.validate_records(result,manifest,'timing')
    assert result['passed'] and not result['sampled'] and result['runtime'] == '.NET 10.0.8' and result['processor_count'] == 1
    assert result['core_sha256'] == built['core']['sha256'] and result['data_sha256'] == spec['data']['sha256']
    assert result['runner_sha256'] == built['consumer']['sha256']
    assert result['manifest_sha256'] == pin(APP/'collected/manifests/current-parakeet.json')['sha256']
    previous = read(ROOT/'artifacts/parakeet-managed-phase-amd-20260924/capture-collected/control/result.json')
    observations = []; geometry = Counter(); offsets = Counter()
    for index, request in enumerate(result['records']):
        assert request == read(folder/'phase'/f'{index:03}.json')
        assert request['result'] == previous['records'][index]['result']
        assert request['thread_id'] == run['ready']['thread_id']
        record = read(folder/'phase'/f'layout-{index:03}.json')
        assert (record['Name'],record['Pass']) == (request['name'],request['pass'])
        assert len(record['Records']) == 24
        frames = request['result']['encoded_frames']
        for layer,row in enumerate(record['Records']):
            spans = qualify(row,frames,layer)
            observations.append(dict(request=index,layer=layer,frames=frames,spans=spans,array_offset=row['ArrayOffset']))
            geometry[frames] += 1; offsets[row['ArrayOffset']] += 1
    assert len(observations) == 80*24
    analysis = dict(passed=True,requests=80,layouts=len(observations),regions_per_layout=8,
        frame_layout_counts=dict(sorted(geometry.items())),array_offset_counts=dict(sorted(offsets.items())),
        exact_public_results=True,data_unchanged=True,all_parents_dense_row_major=True,
        all_slices_unit_step=True,all_copy_regions_in_bounds=True,resources=resource_rows,
        corpus_seconds=sum(r['seconds'] for r in result['records'] if r['phase']=='measured')/3,
        diagnostic_only=True,observations=observations)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),build_review=pin(BASE/'build-review.json'),
        collection=pin(folder/'capture-collection.json'),transfer=pin(BASE/'capture-transfer.json'),auditor=pin(__file__),
        terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for p,b in run['members'].items()]))
    print(json.dumps({k:v for k,v in analysis.items() if k != 'observations'}))


if __name__ == '__main__': main()
