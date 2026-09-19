"""Recreate the pinned WAV/PCM/Whisper-feature inputs used by the audio lanes."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import math
import subprocess
import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import firwin, resample_poly
import transformers
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def convert(samples, source, target):
    if source == target:
        return samples.copy()
    divisor = math.gcd(source, target)
    up, down = target // divisor, source // divisor
    factor = max(up, down)
    taps = firwin(64 * factor + 1, 0.94 / factor, window=('kaiser', 8.6))
    return resample_poly(samples.astype(np.float64), up, down, window=taps).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert (np.__version__, scipy.__version__, transformers.__version__) == ('2.2.4', '1.16.3', '5.16.1')
    pins_path = Path(__file__).with_name('recordings.json')
    pins = json.loads(pins_path.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    originals = {}
    for name, entry in pins['recordings'].items():
        recording = args.recordings / entry['path']
        assert sha(recording) == entry['sha256'], recording
        raw = args.output / (name + '-original.f32')
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(recording), '-ac', '1', '-ar', '16000', '-f', 'f32le', str(raw)]
        subprocess.run(command, check=True)
        samples = np.fromfile(raw, dtype='<f4')
        assert np.isfinite(samples).all()
        pcm = args.output / (name + '-original.npy')
        np.save(pcm, samples, allow_pickle=False)
        assert sha(pcm) == entry['pcm_sha256'], 'Decoded PCM differs: ' + name
        originals[name] = (samples, entry['language'])
    originals['silence'] = (np.zeros(16000, np.float32), 'en')
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400, dither=0.0)
    cases = []
    for name, original, rate, stereo, integer, limit in (
            ('english-16k', 'librispeech', 16000, False, False, 64),
            ('french-44k-stereo', 'french', 44100, True, False, 64),
            ('jfk-48k-stereo', 'jfk', 48000, True, True, 64),
            ('silence-32k', 'silence', 32000, False, False, 64),
            ('jfk-token-limit', 'jfk', 16000, False, False, 1)):
        samples, language = originals[original]
        recording = convert(samples, 16000, rate)
        if stereo:
            recording = np.stack([recording * np.float32(.75), recording * np.float32(1.25)], axis=1)
        if integer:
            recording = np.rint(np.clip(recording, -1, 32767 / 32768) * 32768).astype(np.int16)
        wav = args.output / (name + '.wav')
        wavfile.write(wav, rate, recording)
        read_rate, decoded = wavfile.read(wav)
        assert read_rate == rate and np.array_equal(recording, decoded)
        decoded = decoded.astype(np.float64)
        if integer:
            decoded /= 32768
        if stereo:
            decoded = decoded.mean(axis=1)
        converted = convert(decoded.astype(np.float32), rate, 16000)
        padded = np.zeros((1, 480000), np.float32)
        assert len(converted) <= 480000
        padded[0, :len(converted)] = converted
        features = extractor._np_extract_fbank_features(padded, 'cpu')
        pcm = args.output / (name + '-pcm.npy')
        mel = args.output / (name + '-features.npy')
        np.save(pcm, converted, allow_pickle=False)
        np.save(mel, features, allow_pickle=False)
        for kind, path in [('wav', wav), ('pcm', pcm), ('features', mel)]:
            assert sha(path) == pins['cases'][name][kind + '_sha256'], name + ' ' + kind
        cases.append(dict(name=name, language=language, max_new_tokens=limit, samples=len(converted),
            pcm=pcm.name, pcm_sha256=sha(pcm), features=mel.name, features_sha256=sha(mel),
            wav=wav.name, wav_sha256=sha(wav), source_rate=rate, channels=2 if stereo else 1,
            source=dict(original=original, pins_sha256=sha(pins_path), transform='Independent rational Kaiser resampling, stereo gain and optional PCM16 quantization')))
        print(name, len(converted), 'all three asset hashes match', flush=True)
    manifest = dict(sample_rate=16000, numpy=np.__version__, scipy=scipy.__version__, transformers=transformers.__version__,
        generator_sha256=sha(Path(__file__)), pins_sha256=sha(pins_path),
        extractor_source_sha256=sha(Path(inspect.getfile(WhisperFeatureExtractor))),
        resampling_source_sha256=sha(Path(inspect.getfile(resample_poly))), wav_source_sha256=sha(Path(inspect.getfile(wavfile))),
        ffmpeg_version=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0], cases=cases)
    (args.output / 'audio.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
