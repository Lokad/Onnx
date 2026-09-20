"""Select pinned recordings before inference; preserve both decodes and noise recipe."""
from pathlib import Path
import argparse
import hashlib
import io
import subprocess
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from common import REVISION, LOCALES, BANDS, read, write_new, pin, sha, select, normalize, variants


def metadata(path):
    rows = pq.read_table(path, columns=['id', 'num_samples', 'path', 'gender', 'raw_transcription', 'transcription']).to_pylist()
    result = []
    for index, row in enumerate(rows):
        filename = Path(row['path'].replace('\\', '/')).name
        assert filename and row['gender'] in (0, 1) and row['num_samples'] > 0 and normalize(row['raw_transcription'])
        result.append(dict(row=index, id=row['id'], samples=row['num_samples'], filename=filename, gender=row['gender'],
                           reference_text=row['raw_transcription'], dataset_transcription=row['transcription']))
    return result


def prepare(dataset, output):
    pins_path = Path(__file__).with_name('dataset.json')
    pins = read(pins_path)
    assert pins['revision'] == REVISION and pins['locales'] == [list(v) for v in LOCALES]
    assert not output.exists()
    for name, wanted in pins['files'].items():
        assert pin(dataset / name) == {k: wanted[k] for k in ('bytes', 'sha256')}, name
    inventories = []
    for locale, language in LOCALES:
        path = dataset / f'parquet-data/{locale}/test-00000-of-00001.parquet'
        inventory = metadata(path)
        inventories.append(dict(locale=locale, language=language, inventory=inventory, selected=select(inventory)))
    output.mkdir(parents=True)
    write_new(output / 'selection.json', dict(schema=1, pins_sha256=sha(pins_path), bands=BANDS, locales=inventories))
    # The full selection is now frozen, before any audio transform or recognizer.
    cases = []
    for group in inventories:
        locale, language = group['locale'], group['language']
        wanted = {r['row']: r for r in group['selected']}
        blobs = {}
        index = 0
        parquet = pq.ParquetFile(dataset / f'parquet-data/{locale}/test-00000-of-00001.parquet')
        for batch in parquet.iter_batches(columns=['audio'], batch_size=16):
            for item in batch.to_pylist():
                if index in wanted:
                    value = item['audio']
                    assert Path(value['path'].replace('\\', '/')).name == wanted[index]['filename']
                    blobs[index] = value['bytes']
                index += 1
        assert set(blobs) == set(wanted) and index == len(group['inventory'])
        for selected in group['selected']:
            stem = locale + '-' + Path(selected['filename']).stem
            source = output / (stem + '.source' + Path(selected['filename']).suffix)
            source.write_bytes(blobs[selected['row']])
            pcm, rate = sf.read(io.BytesIO(blobs[selected['row']]), dtype='float32')
            assert rate == 16000 and pcm.shape == (selected['samples'],) and np.isfinite(pcm).all()
            decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1'],
                                     check=True, capture_output=True).stdout
            assert decoded == pcm.tobytes(), stem
            original_path = output / (stem + '.original.npy')
            np.save(original_path, pcm, allow_pickle=False)
            clean, noisy, noise = variants(pcm, locale, selected['filename'])
            for condition, values in [('clean', clean), ('noise10db', noisy)]:
                name = stem + '-' + condition
                pcm_path = output / (name + '.npy')
                np.save(pcm_path, values, allow_pickle=False)
                cases.append(dict(name=name, locale=locale, language=language, condition=condition, source_id=selected['id'],
                    source_row=selected['row'], gender=selected['gender'], samples=len(values), reference_text=selected['reference_text'],
                    pcm=pcm_path.name, pcm_sha256=sha(pcm_path), original=original_path.name, original_sha256=sha(original_path),
                    source=source.name, source_sha256=sha(source), noise=noise, decoders_identical=True))
    assert len(cases) == 40 and len({c['name'] for c in cases}) == 40
    assert len({c['original_sha256'] for c in cases}) == 20
    write_new(output / 'audio.json', dict(schema=1, protocol='multilingual-noise-asr-v2', sample_rate=16000,
        pins_sha256=sha(pins_path), selection_sha256=sha(output/'selection.json'), cases=cases))
    print('Prepared', len(cases), 'cases;', sum(c['samples'] for c in cases) / 16000, 'audio seconds; no inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset.resolve(), args.output.resolve())
