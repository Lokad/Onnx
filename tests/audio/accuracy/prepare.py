"""Freeze a prospectively selected, labeled LibriSpeech diagnostic subset.

Downloads nothing. The pinned parquet must already exist. This is not the full
LibriSpeech test benchmark and does not establish conversational accuracy.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import io
import json
import subprocess
# Import the frontend before Arrow loads its Windows native libraries. Only the
# explicitly selected NumPy feature path below is used for these fixtures.
import transformers
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
import numpy as np
import pyarrow
import pyarrow.parquet as pq
import soundfile as sf


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select(inventory, pins):
    ids = [v['id'] for v in inventory]
    if len(ids) != len(set(ids)) or len(ids) != pins['rows']:
        raise ValueError('Dataset coverage or duplicate utterance ID')
    selected = []
    speakers = sorted({v['speaker_id'] for v in inventory} - set(pins['excluded_speakers']))
    eligible = 0
    for speaker in speakers:
        choices = []
        for low, high in pins['duration_bands_seconds']:
            candidates = [v for v in inventory if v['speaker_id'] == speaker
                          and low * 16000 <= v['samples'] <= high * 16000]
            if not candidates:
                break
            choices.append(min(candidates, key=lambda v: v['id']))
        if len(choices) == len(pins['duration_bands_seconds']):
            selected.extend(choices)
            eligible += 1
        if eligible == pins['speakers']:
            break
    if eligible != pins['speakers'] or len({v['id'] for v in selected}) != len(selected):
        raise ValueError('Insufficient eligible speakers or overlapping selections')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (np.__version__, pyarrow.__version__, sf.__version__, transformers.__version__) != (
            '2.2.4', '18.1.0', '0.13.1', '5.16.1'):
        raise ValueError('Reference dependency versions differ')
    pins_path = Path(__file__).with_name('dataset.json')
    pins = json.loads(pins_path.read_text(encoding='utf-8'))
    if args.parquet.stat().st_size != pins['bytes'] or sha(args.parquet) != pins['sha256']:
        raise ValueError('Dataset bytes differ from pinned revision')
    args.output.mkdir(parents=True, exist_ok=False)
    parquet = pq.ParquetFile(args.parquet)
    inventory = []
    for batch in parquet.iter_batches(batch_size=100):
        for row in batch.to_pylist():
            audio = row['audio']['bytes']
            info = sf.info(io.BytesIO(audio))
            if info.samplerate != 16000 or info.channels != 1 or info.format != 'FLAC':
                raise ValueError('Unexpected source audio contract')
            inventory.append(dict(id=row['id'], speaker_id=row['speaker_id'], chapter_id=row['chapter_id'],
                samples=info.frames, text=row['text'], audio_sha256=hashlib.sha256(audio).hexdigest()))
    selected = select(inventory, pins)
    selection = dict(dataset=pins, inventory=inventory, selected=selected)
    selection_path = args.output / 'selection.json'
    selection_path.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    # Freeze selection before feature construction or any recognition outputs.
    wanted = {v['id']: v for v in selected}
    audio_bytes = {}
    for batch in parquet.iter_batches(batch_size=100):
        for row in batch.to_pylist():
            if row['id'] in wanted:
                audio_bytes[row['id']] = row['audio']['bytes']
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000,
        hop_length=160, chunk_length=30, n_fft=400, dither=0.0)
    extractor_sha = sha(Path(inspect.getfile(WhisperFeatureExtractor)))
    if extractor_sha != 'dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb':
        raise ValueError('Whisper feature extractor source differs')
    cases = []
    for item in selected:
        name = item['id']
        raw = audio_bytes[name]
        if hashlib.sha256(raw).hexdigest() != item['audio_sha256']:
            raise ValueError('Selected audio changed')
        flac = args.output / (name + '.flac')
        flac.write_bytes(raw)
        pcm, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
        independent = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(flac),
            '-f', 'f32le', '-acodec', 'pcm_f32le', '-'])
        if rate != 16000 or pcm.shape != (item['samples'],) or pcm.tobytes() != independent:
            raise ValueError('Independent complete PCM decoding differs')
        if not np.isfinite(pcm).all() or not np.any(pcm) or len(pcm) > 480000:
            raise ValueError('Invalid selected recording')
        padded = np.zeros((1, 480000), np.float32)
        padded[0, :len(pcm)] = pcm
        features = extractor._np_extract_fbank_features(padded, 'cpu')
        pcm_path = args.output / (name + '-pcm.npy')
        feature_path = args.output / (name + '-features.npy')
        np.save(pcm_path, pcm, allow_pickle=False)
        np.save(feature_path, features, allow_pickle=False)
        cases.append(dict(name=name, language='en', speaker_id=item['speaker_id'], chapter_id=item['chapter_id'],
            reference_text=item['text'], samples=len(pcm), max_new_tokens=444,
            pcm=pcm_path.name, pcm_sha256=sha(pcm_path), features=feature_path.name,
            features_sha256=sha(feature_path), flac=flac.name, flac_sha256=sha(flac)))
        print(name, item['speaker_id'], len(pcm) / 16000, 'independent PCM match', flush=True)
    manifest = dict(schema=1, scope='Fixed 20-utterance, 10-speaker English read-speech diagnostic',
        sample_rate=16000, dataset=pins, pins_sha256=sha(pins_path), selection_sha256=sha(selection_path),
        generator_sha256=sha(Path(__file__)), extractor_source_sha256=extractor_sha,
        numpy=np.__version__, pyarrow=pyarrow.__version__, soundfile=sf.__version__,
        libsndfile=sf.__libsndfile_version__, transformers=transformers.__version__,
        ffmpeg_version=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0], cases=cases)
    (args.output / 'audio.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
