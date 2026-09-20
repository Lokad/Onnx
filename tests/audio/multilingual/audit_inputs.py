"""Reconstruct immutable dataset selection and verify every prepared signal."""
from pathlib import Path
import argparse
import io
import math
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from common import REVISION, LOCALES, BANDS, pin, read, sha, write_new, variants
from prepare import metadata


def check_selection(selection, audio, inventories):
    assert selection['schema'] == audio['schema'] == 1
    assert selection['bands'] == [list(b) for b in BANDS]
    assert audio['protocol'] == 'multilingual-noise-asr-v2' and audio['sample_rate'] == 16000
    assert len(selection['locales']) == len(LOCALES) and len(audio['cases']) == 40
    assert len({c['name'] for c in audio['cases']}) == 40
    expected = []
    for (locale, language), saved, rows in zip(LOCALES, selection['locales'], inventories, strict=True):
        assert saved['locale'] == locale and saved['language'] == language and saved['inventory'] == rows
        assert len({r['filename'] for r in rows}) == len(rows)
        labels = sorted({r['gender'] for r in rows})
        assert labels and set(labels) <= {0, 1}
        used = set()
        chosen = []
        for label in (labels[0], labels[1 % len(labels)]):
            for lo, hi in BANDS:
                candidates = sorted((r for r in rows if r['gender'] == label and lo*16000 <= r['samples'] <= hi*16000),
                                    key=lambda r: (r['id'], r['filename']))
                row = next(r for r in candidates if r['id'] not in used)
                used.add(row['id']); chosen.append(row)
                for condition in ('clean', 'noise10db'):
                    expected.append((locale, language, condition, row))
        assert chosen == saved['selected']
    for case, (locale, language, condition, row) in zip(audio['cases'], expected, strict=True):
        assert (case['locale'], case['language'], case['condition'], case['source_id'], case['source_row'], case['gender'], case['samples'], case['reference_text']) == (
            locale, language, condition, row['id'], row['row'], row['gender'], row['samples'], row['reference_text'])
        assert case['name'] == locale + '-' + Path(row['filename']).stem + '-' + condition
    return expected


def audit(dataset, base):
    pins_path = Path(__file__).with_name('dataset.json'); pins = read(pins_path)
    assert pins['revision'] == REVISION and pins['locales'] == [list(v) for v in LOCALES]
    for name, wanted in pins['files'].items():
        assert pin(dataset/name) == {k:wanted[k] for k in ('bytes','sha256')}, name
    selection, audio = read(base/'selection.json'), read(base/'audio.json')
    assert selection['pins_sha256'] == audio['pins_sha256'] == sha(pins_path)
    assert audio['selection_sha256'] == sha(base/'selection.json')
    inventories = [metadata(dataset/f'parquet-data/{locale}/test-00000-of-00001.parquet') for locale,language in LOCALES]
    expected = check_selection(selection, audio, inventories)
    blobs = {}
    for group in selection['locales']:
        wanted = {r['row']:r for r in group['selected']}
        parquet = pq.ParquetFile(dataset/f"parquet-data/{group['locale']}/test-00000-of-00001.parquet")
        index = 0
        for batch in parquet.iter_batches(batch_size=16, columns=['audio']):
            for row in batch.to_pylist():
                if index in wanted:
                    blobs[(group['locale'], index)] = row['audio']['bytes']
                index += 1
    files = {'selection.json', 'audio.json'}
    for index in range(0, 40, 2):
        clean_case, noisy_case = audio['cases'][index:index+2]
        _, _, _, chosen = expected[index]
        for case in (clean_case, noisy_case):
            for key in ('pcm','original','source'):
                assert Path(case[key]).name == case[key] and sha(base/case[key]) == case[key+'_sha256']
                files.add(case[key])
            assert case['decoders_identical'] is True
        assert clean_case['original'] == noisy_case['original'] and clean_case['source'] == noisy_case['source']
        assert (base/clean_case['source']).read_bytes() == blobs[(clean_case['locale'], clean_case['source_row'])]
        original = np.load(base/clean_case['original'], allow_pickle=False)
        decoded, rate = sf.read(io.BytesIO(blobs[(clean_case['locale'], clean_case['source_row'])]), dtype='float32')
        assert rate == 16000 and original.dtype == np.float32 and original.shape == (chosen['samples'],) and original.tobytes() == decoded.tobytes()
        clean = np.load(base/clean_case['pcm'],allow_pickle=False)
        noisy = np.load(base/noisy_case['pcm'],allow_pickle=False)
        assert clean.dtype == noisy.dtype == np.float32 and clean.shape == noisy.shape == original.shape
        assert np.isfinite(clean).all() and np.isfinite(noisy).all() and max(np.abs(clean).max(),np.abs(noisy).max()) <= 1
        a,b,info = variants(original, clean_case['locale'], chosen['filename'])
        assert clean.tobytes() == a.tobytes() and noisy.tobytes() == b.tobytes()
        assert clean_case['noise'] == noisy_case['noise'] == info
        signal = clean.astype(np.float64); noise = noisy.astype(np.float64)-signal
        actual_snr = 10*math.log10(float(signal@signal)/float(noise@noise))
        assert abs(actual_snr-10) < .001
    assert {p.name for p in base.iterdir()} == files
    return dict(schema=1,passed=True,recordings=20,cases=40,audio_seconds=sum(c['samples'] for c in audio['cases'])/16000,
                audio_sha256=sha(base/'audio.json'),selection_sha256=sha(base/'selection.json'),pins_sha256=sha(pins_path),
                files={name:pin(base/name) for name in sorted(files)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','inputs','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    result=audit(args.dataset.resolve(),args.inputs.resolve());write_new(args.output,result)
    print('Independent dataset/selection/signal audit passed:',result['cases'],'cases,',result['audio_seconds'],'seconds.')
