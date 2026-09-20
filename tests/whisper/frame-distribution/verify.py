"""Reconstruct every saved frame metric and region independently from the raw arrays."""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True, type=Path)
    base = parser.parse_args().artifact.resolve()
    assert not (base/'verification.json').exists()
    value = json.loads((base/'observations.json').read_text())
    for name, expected in value['inputs'].items():
        assert pin(ROOT/name) == expected, name
    assert pin(base/'frame-metrics.npz') == value['metrics']
    assert pin(base/'WhisperAudio-original.cs') == value['frontend']
    assert pin(Path(__file__).with_name('analyze.py')) == value['source']
    metrics = np.load(base/'frame-metrics.npz', allow_pickle=False)
    terms = list(value['unique20'])
    assert set(metrics.files) == {term+'__'+key for term in terms for key in ['failed', 'maximum', 'squares']}
    original = ROOT/'artifacts/whisper-input-cross-isolated-20260920'
    aggregate = {size: {term: {region: dict(frames=0, values=0, failed_frames=0, failed_values=0,
                                         max_scaled=0., sum_squares=0.)
                              for region in ['before_end', 'end_boundary', 'after_end']} for term in terms}
                 for size in [20, 21]}
    checked = 0
    for index, row in enumerate(value['rows']):
        cells = {}
        for engine in ['managed', 'native']:
            folder = original/'outputs'/f"{engine}-{index:02}-{row['name']}"
            info = json.loads((folder/'result.json').read_text())
            for record in info['records']:
                cells[record['kind']] = np.fromfile(folder/record['file'], dtype='<f4').astype(np.float64).reshape(1500, 1280)
        tags = []
        for position in range(1500):
            # Enumerate both convolutions' taps, independently of the analyzer's 520 constant.
            features = [(position*2 + second-1) + first-1 for second in range(3) for first in range(3)]
            low, high = min(features)*160-200, max(features)*160+200
            tags.append('before_end' if high <= row['samples'] else 'after_end' if low >= row['samples'] else 'end_boundary')
        tags = np.asarray(tags)
        nn = cells['NN']; denominator = np.where(np.abs(nn) < 1, 1., np.abs(nn))
        pairs = {'original_MM-NN': ('MM', 'NN'), 'engine_MM-NM': ('MM', 'NM'), 'engine_MN-NN': ('MN', 'NN'),
                 'input_MM-MN': ('MM', 'MN'), 'input_NM-NN': ('NM', 'NN')}
        for term in terms:
            if term == 'interaction':
                delta = (cells['MM']-cells['MN'])-(cells['NM']-cells['NN'])
            else:
                first, second = pairs[term]; delta = cells[first]-cells[second]
            scaled = np.abs(delta)/denominator
            failed = np.bincount(np.nonzero(scaled > 1e-4)[0], minlength=1500)
            maximum = np.maximum.reduce(scaled, axis=1)
            squares = np.add.reduceat((delta*delta).ravel(), np.arange(0, delta.size, 1280))
            assert np.array_equal(failed, metrics[term+'__failed'][index])
            assert np.array_equal(maximum, metrics[term+'__maximum'][index])
            assert np.allclose(squares, metrics[term+'__squares'][index], rtol=1e-12, atol=1e-18)
            details = row['terms'][term]
            flat = int(np.argmax(scaled)); worst = details['worst']
            assert (worst['frame'], worst['channel']) == divmod(flat, 1280)
            assert worst['scaled'] == float(scaled.ravel()[flat]) and worst['region'] == tags[worst['frame']]
            assert worst['center_seconds'] == worst['frame']*.02
            for region, expected in details['regions'].items():
                selected = tags == region
                observed = dict(frames=int(np.count_nonzero(selected)), values=int(np.count_nonzero(selected))*1280,
                    failed_frames=int(np.count_nonzero(failed[selected])), failed_values=int(failed[selected].sum()),
                    max_scaled=float(maximum[selected].max(initial=0)), sum_squares=float(squares[selected].sum()))
                for key, number in observed.items():
                    if key == 'sum_squares': assert math.isclose(number, expected[key], rel_tol=1e-12, abs_tol=1e-15)
                    else: assert number == expected[key], (index, term, region, key)
                for size in [20, 21]:
                    if index < size:
                        destination = aggregate[size][term][region]
                        for key, number in observed.items():
                            destination[key] = max(destination[key], number) if key == 'max_scaled' else destination[key]+number
            checked += delta.size
    assert checked == 241920000 and len(value['rows']) == 21
    for size, label in [(20, 'unique20'), (21, 'including_repeat21')]:
        for term in terms:
            for region, expected in value[label][term].items():
                got = aggregate[size][term][region]
                got['failed_value_rate'] = got['failed_values']/got['values'] if got['values'] else 0.
                for key, number in got.items():
                    if key == 'sum_squares': assert math.isclose(number, expected[key], rel_tol=1e-12, abs_tol=1e-15)
                    else: assert number == expected[key], (size, term, region, key)
    for name in metrics.files:
        assert metrics[name].shape == (21, 1500) and np.array_equal(metrics[name][0], metrics[name][20])
    result = dict(passed=True, checked_difference_values=checked, frames=21*1500, terms=6, source=pin(Path(__file__)),
                  observations=pin(base/'observations.json'), metrics=pin(base/'frame-metrics.npz'), inputs=len(value['inputs']),
                  scope='Independent full-array metric and geometry reconstruction; no new inference or acceptance exception')
    with (base/'verification.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
