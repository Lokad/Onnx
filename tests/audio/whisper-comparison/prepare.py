"""Bind existing complete Whisper reference records; do not regenerate old evidence."""
from pathlib import Path
import argparse, hashlib, inspect, json
import importlib.metadata
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
import transformers.audio_utils


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return Path(path).read_text(encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = Path(__file__).resolve().parents[3]; base = root / 'artifacts/asr-labeled-20260919'
    assert not a.output.exists()
    def file(path, digest=None):
        path = Path(path); path = path if path.is_absolute() else root / path
        h = sha(path)
        if digest is not None: assert h == digest, path
        try: name = path.relative_to(root).as_posix()
        except ValueError: name = str(path)
        return dict(path=name, sha256=h, bytes=path.stat().st_size)
    assert sha(base / 'receipt.json') == 'a4855ce825f4d5f1ea931440d031eb3a0aa5b3a0fee1fddbcf4d5b51500754b7'
    receipt = json.loads(read(base / 'receipt.json'))
    for name in ('inputs/audio.json', 'native-whisper/manifest.json'):
        assert sha(base / name) == receipt['files'][name]['sha256']
    audio = json.loads(read(base / 'inputs/audio.json')); native = json.loads(read(base / 'native-whisper/manifest.json'))
    assets = json.loads(read(root / 'tests/whisper/transcription-assets.json'))
    assert native['assets'] == assets and native['audio_manifest_sha256'] == sha(base / 'inputs/audio.json')
    models = {name: file(root / 'models/whisper-large-v3-turbo' / name, pin['sha256']) for name, pin in assets['files'].items()}
    versions = {'numpy': '2.2.4', 'onnxruntime': '1.29.0', 'transformers': '5.16.1', 'tokenizers': '0.23.2'}
    for name, version in versions.items(): assert importlib.metadata.version(name) == version
    upstream = dict(extractor=file(inspect.getfile(WhisperFeatureExtractor), audio['extractor_source_sha256']),
                    audio_utils=file(inspect.getfile(transformers.audio_utils)),
                    reference_generator=file(root / 'tests/whisper/generate_transcription_reference.py'))
    cases = []
    assert len(audio['cases']) == len(native['cases']) == 20
    for c, n in zip(audio['cases'], native['cases'], strict=True):
        assert c['name'] == n['name'] and c['pcm_sha256'] == n['pcm_sha256'] and c['language'] == 'en' and c['max_new_tokens'] == 444
        expected = dict(text=n['text'], token_ids=n['tokens'], stop_reason=n['stop_reason'], skipped_as_no_speech=n['skipped_as_no_speech'])
        assert len(n['tokens']) == len(n['steps']) and [s['token'] for s in n['steps']] == n['tokens']
        cases.append(dict(name=c['name'], samples=c['samples'], pcm=file(base / 'inputs' / c['pcm'], c['pcm_sha256']),
                          features=file(base / 'inputs' / c['features'], c['features_sha256']), expected=expected))
    result = dict(schema=1, family='whisper', models=models, assets=assets, upstream=upstream, versions=versions,
                  reference=file(base / 'native-whisper/manifest.json'), audio_manifest=file(base / 'inputs/audio.json'),
                  cases=cases, warmup_passes=1, measured_passes=3, language='en', max_new_tokens=444)
    a.output.mkdir(parents=True)
    with (a.output / 'whisper.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print('Prepared', len(cases), 'clips,', sum(c['samples'] for c in cases) / 16000, 'seconds', sha(a.output / 'whisper.json'))


if __name__ == '__main__': main()
