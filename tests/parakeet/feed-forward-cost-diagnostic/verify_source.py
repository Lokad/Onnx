"""Check the retained diagnostic edits, exact original bytes, and consumer reuse."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'artifacts/parakeet-feed-forward-cost-source-20260925'
OUT = ROOT / 'artifacts/parakeet-feed-forward-cost-observer-source-20260925'


def pin(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    assert not OUT.exists(), 'Preserve the completed source review'
    review = read(SOURCE / 'review.json')
    assert review['passed'] and not review['compiled'] and not review['executed']
    assert review['generator'] == pin(HERE / 'source.py')
    assert review['patch'] == pin(SOURCE / 'diagnostic.patch')
    assert review['added_source'] == pin(SOURCE / 'FeedForwardCostStages.cs')
    endings = {}
    for name, row in review['files'].items():
        original = ROOT / 'src/Lokad.Onnx' / name
        assert pin(original) == row['original'] and pin(SOURCE / name) == row['diagnostic']
        restored = (SOURCE / name).read_text(encoding='utf8')
        for edit in reversed(row['edits']):
            assert restored.count(edit['after']) == 1
            restored = restored.replace(edit['after'], edit['before'])
        data = original.read_bytes()
        crlf = b'\r\n' in data
        reproduced = restored.replace('\n', '\r\n').encode() if crlf else restored.encode()
        assert reproduced == data, name
        endings[name] = 'CRLF' if crlf else 'LF'
    transcriber = ROOT / 'src/Lokad.Onnx.Data/ParakeetTranscriber.cs'
    original = transcriber.read_text(encoding='utf8')
    anchor = ('    static IReadOnlyDictionary<string, ITensor> Execute(GraphExecution context, Dictionary<string, ITensor> feeds)\n'
              '    {\n')
    addition = '        using var observation = ParakeetPhaseProbe.Enter(context);\n'
    assert original.count(anchor) == 1 and addition not in original
    modified = original.replace(anchor, anchor + addition)
    assert modified.replace(addition, '') == original
    certification = ROOT / 'tests/parakeet/slice-dense-conversion-results/observer-20260925.json'
    prior = read(certification)
    assert prior['passed'] and prior['all_original_bodies_flags_and_surface_exact']
    assert pin(certification)['sha256'] == '11dad08be4a439f63d9d5dfff4ff2169cb31ea9276d6284657cd1f25636a9bd0'
    runtime = ROOT / prior['observer_runtime']
    assert pin(runtime / 'SampledAudio.dll') == prior['consumer']
    consumer_source = ROOT / 'artifacts/parakeet-slice-materialization-profile-amd-20260924/bundle/consumer-source'
    consumer = (consumer_source / 'Program.cs').read_text(encoding='utf8')
    assert 'Environment.GetEnvironmentVariable("PARAKEET_PHASE_CORE_SHA")' in consumer
    assert 'Environment.GetEnvironmentVariable("PARAKEET_PHASE_DATA_SHA")' in consumer
    assert 'PhaseConsumer.Save(output,records.Count,c.Name,pass);' in consumer
    expected_graphs = ROOT / 'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924/capture-collected/wall/graphs.json'
    assert pin(expected_graphs) == prior['graph_metadata']
    graphs = read(expected_graphs)
    encoder = graphs['encoder-model.onnx']
    targets = [n for n in encoder['nodes'] if n['op'] == 'MatMul' and '/feed_forward' in n['name']]
    assert len(encoder['nodes']) == 2856 and len(targets) == 96
    assert all(len(n['inputs']) == 2 and n['constant_inputs'][1]['dims'] in [[1024, 4096], [4096, 1024]] for n in targets)
    OUT.mkdir()
    with (OUT / 'ParakeetTranscriber.cs').open('x', encoding='utf8', newline='\n') as stream:
        stream.write(modified)
    with (OUT / 'PhaseProbe.cs').open('xb') as stream:
        stream.write((HERE / 'PhaseProbe.cs.txt').read_bytes())
    result = dict(passed=True, scope='source-and-existing-consumer-review-only',
                  compiled=False, executed=False, core_source_review=pin(SOURCE / 'review.json'),
                  exact_original_byte_restoration=True, original_line_endings=endings,
                  original_transcriber=pin(transcriber), diagnostic_transcriber=pin(OUT / 'ParakeetTranscriber.cs'),
                  observer=pin(OUT / 'PhaseProbe.cs'), consumer=prior['consumer'],
                  consumer_certification=pin(certification), expected_graphs=pin(expected_graphs),
                  encoder_nodes=2856, feed_forward_nodes=96,
                  consumer_rebuild_required=False, data_and_core_build_required=True,
                  modes=['clock', 'stages', 'markers'],
                  required_consumer_environment=dict(PARAKEET_PHASE_MODE='phase',
                      PARAKEET_PHASE_CORE_SHA='exact assigned Core hash', PARAKEET_PHASE_DATA_SHA='exact diagnostic Data hash'),
                  reviewer=pin(Path(__file__)))
    with (OUT / 'review.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, review=pin(OUT / 'review.json'), consumer_rebuild=False,
                         source_only=True, encoder_nodes=2856, targets=96)))


if __name__ == '__main__':
    main()
