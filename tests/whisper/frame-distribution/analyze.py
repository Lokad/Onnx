"""Describe every saved Whisper discrepancy by encoder position; never exclude padding."""
from pathlib import Path
import argparse, hashlib, json, math, subprocess
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[3]
ORIGIN = '513f355ef7a36959c6c6454fbb65da33d3feb20585104267bf500113b1da6054'
CORPUS = 'a4855ce825f4d5f1ea931440d031eb3a0aa5b3a0fee1fddbcf4d5b51500754b7'
REGIONS = ['before_end', 'end_boundary', 'after_end']
TERMS = ['original_MM-NN', 'engine_MM-NM', 'engine_MN-NN', 'input_MM-MN', 'input_NM-NN', 'interaction']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def regions(samples):
    centers = np.arange(1500, dtype=np.int64) * 320
    before = centers + 520 <= samples
    after = centers - 520 >= samples
    masks = [before, ~(before | after), after]
    assert np.all(np.sum(masks, axis=0) == 1)
    return dict(zip(REGIONS, masks))


def describe(difference, denominator, mask):
    selected = difference[mask]
    scaled = np.abs(selected) / denominator[mask]
    failed = scaled > 1e-4
    return dict(frames=int(mask.sum()), values=int(selected.size),
                failed_frames=int(np.any(failed, axis=1).sum()), failed_values=int(failed.sum()),
                max_scaled=float(scaled.max(initial=0)), sum_squares=float(np.sum(selected * selected)))


def aggregate(rows):
    result = {}
    for term in TERMS:
        result[term] = {}
        for region in REGIONS:
            values = [row['terms'][term]['regions'][region] for row in rows]
            counts = {key: sum(v[key] for v in values)
                      for key in ['frames', 'values', 'failed_frames', 'failed_values', 'sum_squares']}
            counts['max_scaled'] = max(v['max_scaled'] for v in values)
            counts['failed_value_rate'] = counts['failed_values'] / counts['values'] if counts['values'] else 0.
            result[term][region] = counts
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    output = parser.parse_args().artifact.resolve()
    assert not output.exists(), 'Analysis destination is single-use'
    source = ROOT / 'artifacts/whisper-input-cross-isolated-20260920'
    corpus = ROOT / 'artifacts/asr-labeled-20260919'
    assert pin(source / 'closed.json')['sha256'] == ORIGIN
    closed = read(source / 'closed.json')
    inputs = {}

    def bind(path, expected=None):
        actual = pin(path)
        if expected is not None:
            assert actual == expected, path
        inputs[path.relative_to(ROOT).as_posix()] = actual
        return actual

    bind(source / 'closed.json')
    for name in ['manifest.json', 'audit.json']:
        bind(source / name, closed['files'][name])
    manifest, previous = read(source / 'manifest.json'), read(source / 'audit.json')
    assert previous['structural_passed'] and previous['arrays'] == 84 and len(manifest['requests']) == 21
    bind(corpus / 'receipt.json', manifest['files'][(corpus / 'receipt.json').relative_to(ROOT).as_posix()])
    assert pin(corpus / 'receipt.json')['sha256'] == CORPUS
    corpus_receipt = read(corpus / 'receipt.json')
    bind(corpus / 'inputs/audio.json', corpus_receipt['files']['inputs/audio.json'])
    audio = read(corpus / 'inputs/audio.json')
    assert audio['sample_rate'] == 16000 and len(audio['cases']) == 20
    assert [r['name'] for r in manifest['requests']] == [c['name'] for c in audio['cases']] + [audio['cases'][0]['name']]

    model_path = ROOT / manifest['model']
    bind(model_path, manifest['files'][manifest['model']])
    model = onnx.load(model_path, load_external_data=False)
    nodes = {node.name: node for node in model.graph.node}
    geometry = []
    for name, stride in [('/conv1/Conv', 1), ('/conv2/Conv', 2)]:
        node = nodes[name]
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert node.op_type == 'Conv' and attributes['kernel_shape'] == [3]
        assert attributes['strides'] == [stride] and attributes['pads'] == [1, 1]
        assert attributes['dilations'] == [1] and attributes['group'] == 1
        geometry.append(dict(name=name, attributes=attributes))
    frontend = subprocess.check_output(['git', 'show', 'c6bf781:src/Lokad.Onnx.Data/WhisperAudio.cs'], cwd=ROOT)
    for token in [b'SampleRate = 16000', b'WindowSize = 400', b'HopSize = 160', b'Frames = 3000']:
        assert token in frontend
    output.mkdir(parents=True)
    (output / 'WhisperAudio-original.cs').write_bytes(frontend)
    rows, metrics = [], {term + '__' + key: [] for term in TERMS for key in ['failed', 'maximum', 'squares']}
    for index, request in enumerate(manifest['requests']):
        case = audio['cases'][index % 20]
        pcm_path = corpus / 'inputs' / case['pcm']
        bind(pcm_path, corpus_receipt['files'][pcm_path.relative_to(corpus).as_posix()])
        pcm = np.load(pcm_path, allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        cells = {}
        for engine in ['managed', 'native']:
            folder = source / 'outputs' / f"{engine}-{index:02}-{request['name']}"
            bind(folder / 'result.json', closed['files'][(folder / 'result.json').relative_to(source).as_posix()])
            result = read(folder / 'result.json')
            assert result['complete'] and result['request_index'] == index and len(result['records']) == 2
            for record in result['records']:
                path = folder / record['file']
                bind(path, closed['files'][path.relative_to(source).as_posix()])
                assert record['shape'] == [1, 1500, 1280] and record['name'] == request['name']
                assert record['kind'] not in cells
                values = np.fromfile(path, dtype='<f4')
                assert values.size == 1920000 and np.isfinite(values).all()
                cells[record['kind']] = values.reshape(1500, 1280).astype(np.float64)
        assert set(cells) == {'MM', 'MN', 'NM', 'NN'}
        mm, mn, nm, nn = [cells[key] for key in ['MM', 'MN', 'NM', 'NN']]
        differences = dict(zip(TERMS, [mm-nn, mm-nm, mn-nn, mm-mn, nm-nn, (mm-mn)-(nm-nn)]))
        denominator = np.maximum(1., np.abs(nn))
        masks = regions(case['samples'])
        row = dict(request=index, name=request['name'], samples=case['samples'], seconds=case['samples']/16000, terms={})
        for term, difference in differences.items():
            scaled = np.abs(difference) / denominator
            failed = (scaled > 1e-4).sum(axis=1)
            maximum = scaled.max(axis=1)
            squares = np.sum(difference*difference, axis=1)
            old = previous['rows'][index]['terms'][term]
            assert int(failed.sum()) == old['failed_values'] and float(maximum.max()) == old['max_scaled']
            assert math.isclose(float(squares.sum()), old['sum_squares'], rel_tol=1e-12, abs_tol=1e-15)
            frame, channel = [int(v) for v in np.unravel_index(np.argmax(scaled), scaled.shape)]
            partitions = {name: describe(difference, denominator, mask) for name, mask in masks.items()}
            assert sum(v['failed_values'] for v in partitions.values()) == old['failed_values']
            assert sum(v['values'] for v in partitions.values()) == 1920000
            assert math.isclose(sum(v['sum_squares'] for v in partitions.values()), old['sum_squares'], rel_tol=1e-12, abs_tol=1e-15)
            row['terms'][term] = dict(regions=partitions, worst=dict(frame=frame, channel=channel,
                center_seconds=frame*.02, region=next(name for name, mask in masks.items() if mask[frame]),
                scaled=float(scaled[frame, channel])))
            for key, value in [('failed', failed), ('maximum', maximum), ('squares', squares)]:
                metrics[term+'__'+key].append(value)
        rows.append(row)
    metrics = {key: np.stack(values) for key, values in metrics.items()}
    assert all(value.shape == (21, 1500) and np.array_equal(value[0], value[20]) for value in metrics.values())
    np.savez_compressed(output / 'frame-metrics.npz', **metrics)
    observations = dict(schema=1, scope=__doc__, source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        numpy=np.__version__, onnx=onnx.__version__, source=pin(Path(__file__)), origin=pin(source/'closed.json'), corpus=pin(corpus/'receipt.json'),
        inputs=inputs, geometry=dict(sample_rate=16000, feature_hop=160, fft_window=400, convolutions=geometry, encoder_center_samples=320,
            footprint_half_width=520, semantics='Pre-attention local input footprint; attention and log-mel floor are global. No output exclusion.'),
        rows=rows, unique20=aggregate(rows[:20]), including_repeat21=aggregate(rows), metrics=pin(output/'frame-metrics.npz'),
        frontend=pin(output/'WhisperAudio-original.cs'), previous_totals_matched=True, repeat_metrics_exact=True)
    write(output / 'observations.json', observations)
    print(json.dumps(dict(unique20=observations['unique20'], files_bound=len(inputs)), indent=2))


if __name__ == '__main__':
    main()
