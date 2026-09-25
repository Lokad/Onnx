"""Publish the admitted cost split and a read-only initializer ownership census."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-feed-forward-cost-joint-20260925'
REV = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    target = OUT / 'costs-20260925.json'
    assert not target.exists()
    closure = read(BASE / 'closed.json'); analysis = read(BASE / 'analysis.json')
    assert pin(BASE / 'closed.json')['sha256'] == 'd7b79da277b3f6017be67478e67d52c753dc0c2ffdb6d0f31ee27a84f6e9bcab'
    assert closure['passed'] and closure['usable_for_candidate_selection']
    assert closure['analysis'] == pin(BASE / 'analysis.json')
    for name, wanted in closure['files'].items():
        assert pin(BASE / name) == wanted
    assert analysis['usable_for_candidate_selection'] and not analysis['release_admitted'] and not analysis['overhead_subtracted']
    assert all(r['passed'] for r in analysis['controls']['repeatability'] + analysis['controls']['observer_effects'])
    stages = Counter()
    for row in analysis['families']:
        stages.update(row['stages'])
    total = sum(r['complete_seconds'] for r in analysis['families'])
    native = sum(r['earlier_ort_seconds'] for r in analysis['families'])
    graph_path = ROOT / 'artifacts/parakeet-feed-forward-cost-amd-20260925/capture-collected/clock/graphs.json'
    graphs = read(graph_path); graph = graphs['encoder-model.onnx']; uses = defaultdict(list)
    assert len(graph['nodes']) == 2856 and all(n['op'] != 'If' for n in graph['nodes'])
    for node in graph['nodes']:
        for index, name in enumerate(node['inputs']):
            uses[name].append(dict(node=node['name'], op=node['op'], input=index))
    projections = [n for n in graph['nodes'] if n['op'] == 'MatMul' and '/feed_forward' in n['name']]
    weights = {n['inputs'][1]:n['constant_inputs'][1] for n in projections}
    assert len(projections) == len(weights) == 96
    assert Counter(tuple(w['dims']) for w in weights.values()) == {(1024,4096):48, (4096,1024):48}
    assert all(w['type'] == 'Float' and w['name'] == name for name,w in weights.items())
    assert all(len(uses[n]) == 1 and uses[n][0]['op'] == 'MatMul' and uses[n][0]['input'] == 1 for n in weights)
    payload = sum(w['dims'][0] * w['dims'][1] * 4 for w in weights.values())
    assert payload == 1610612736 and graph['maximum_packed_bytes'] == graph['retained_packed_bytes'] == 268435456
    native_path = 'onnxruntime/core/framework/session_state.cc'
    blob = subprocess.check_output(['git', '-C', str(ROOT/'external/onnxruntime'), 'show', REV+':'+native_path])
    source = blob.decode()
    needles = ['--constant_initializers_use_count[input_name] == 0',
               'st->initialized_tensors_.erase(ort_value_idx);', 'constant_initialized_tensors.erase(ort_value_idx);']
    assert all(n in source for n in needles)
    managed_path = ROOT / 'src/Lokad.Onnx/GraphPacking.cs'
    managed = managed_path.read_text()
    assert all(n in managed for n in ['ITensor SourceRef,', 'float[] SourceArray,',
        'new float[(long)n * k]', 'graph.Initializers[packedName] = packed;'])
    result = dict(passed=True, diagnostic_only=True, release_admitted=False, new_implementation_selected=False,
        closure=pin(BASE/'closed.json'), analysis=pin(BASE/'analysis.json'),
        corpus_seconds={k:v['corpus_seconds'] for k,v in analysis['roles'].items()},
        controls=analysis['controls'], families=analysis['families'], stages=dict(stages),
        complete_feed_forward_seconds=total, earlier_ort_seconds=native, profile_gap_seconds=total-native,
        weight_packing_seconds=stages['WeightPacking'], multiplication_seconds=stages['PreparedMultiplication']+stages['PackedMultiplication'],
        packing_fraction_of_diagnostic_gap=stages['WeightPacking']/(total-native),
        scale_seconds=sum(r['scale_seconds'] for r in analysis['families']),
        transition_seconds=sum(r['transition_residual_seconds'] for r in analysis['families']),
        native_profile_is_earlier=True, no_saving_promised=True, no_overhead_subtracted=True,
        graph_metadata=pin(graph_path), feed_forward_weights=[dict(**v, consumers=uses[k]) for k,v in weights.items()],
        feed_forward_payload_bytes=payload, encoder_packed_limit=graph['maximum_packed_bytes'],
        graph_use_census_is_not_a_managed_heap_dump=True, originals_not_proven_removable=True,
        native_release_source=dict(revision=REV, path=native_path, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
            git_object=subprocess.check_output(['git','-C',str(ROOT/'external/onnxruntime'),'rev-parse',REV+':'+native_path],text=True).strip(),
            lines={n:source[:source.index(n)].count('\n')+1 for n in needles}),
        managed_sources={str(p.relative_to(ROOT)).replace('\\','/'):pin(p) for p in [managed_path,
            ROOT/'src/Lokad.Onnx/GraphExecution.cs', ROOT/'src/Lokad.Onnx/Model.cs', ROOT/'src/Lokad.Onnx.Data/ParakeetTranscriber.cs']},
        next_question='Can immutable graph-owned weights retain one prepared payload instead of source plus clone, while preserving fallback, replacement and ownership contracts?',
        publisher=pin(Path(__file__)))
    with target.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(published=pin(target), packing_seconds=stages['WeightPacking'],
        profile_gap_seconds=total-native, weights=len(weights), payload_bytes=payload)))


if __name__ == '__main__':
    main()
