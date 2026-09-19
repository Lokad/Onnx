"""Audit fixed-subset provenance and report native/managed WER and CER.

All errors are retained. Corpus rates use total edit errors / total reference
length, never a mean of utterance rates. No accuracy threshold is fitted here.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import unicodedata
import jiwer
import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize(text):
    value = unicodedata.normalize('NFKC', text).upper()
    result = []
    for i, char in enumerate(value):
        keep = char.isalnum() or (char == "'" and i > 0 and i + 1 < len(value)
                                 and value[i - 1].isalnum() and value[i + 1].isalnum())
        result.append(char if keep else ' ')
    return ' '.join(''.join(result).split())


def distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(row[j] + 1, current[-1] + 1, row[j - 1] + (left != right)))
        row = current
    return row[-1]


def metrics(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        raise ValueError('Human transcript has no scored characters')
    words = jiwer.process_words(reference, hypothesis)
    chars = jiwer.process_characters(reference, hypothesis)
    word_errors = distance(reference.split(), hypothesis.split())
    char_errors = distance(reference, hypothesis)
    if word_errors != words.substitutions + words.deletions + words.insertions or char_errors != chars.substitutions + chars.deletions + chars.insertions:
        raise ValueError('Independent edit distance differs from JiWER')
    return dict(reference=reference, hypothesis=hypothesis, reference_words=len(reference.split()),
        reference_characters=len(reference), word_errors=word_errors, character_errors=char_errors,
        substitutions=words.substitutions, deletions=words.deletions, insertions=words.insertions,
        word_error_rate=word_errors / len(reference.split()), character_error_rate=char_errors / len(reference),
        normalized_equal=reference == hypothesis)


def validate_selection(audio, selection, pins):
    if audio['dataset'] != pins or selection['dataset'] != pins or audio['schema'] != 1 or audio['sample_rate'] != 16000:
        raise ValueError('Dataset/schema identity differs')
    inventory = selection['inventory']
    if len(inventory) != pins['rows'] or len({r['id'] for r in inventory}) != len(inventory):
        raise ValueError('Inventory coverage or duplicate IDs')
    chosen = []
    for speaker in sorted({r['speaker_id'] for r in inventory} - set(pins['excluded_speakers'])):
        bands = [[r for r in inventory if r['speaker_id'] == speaker and low * 16000 <= r['samples'] <= high * 16000]
                 for low, high in pins['duration_bands_seconds']]
        if all(bands):
            chosen.extend(sorted(band, key=lambda r: r['id'])[0] for band in bands)
        if len(chosen) == pins['speakers'] * len(pins['duration_bands_seconds']):
            break
    if selection['selected'] != chosen or len(chosen) != 20 or len(audio['cases']) != 20:
        raise ValueError('Prospective selection differs')
    if len({c['name'] for c in audio['cases']}) != 20 or len({c['pcm_sha256'] for c in audio['cases']}) != 20:
        raise ValueError('Duplicate recording')
    for case, original in zip(audio['cases'], chosen, strict=True):
        if (case['name'], case['speaker_id'], case['chapter_id'], case['samples'], case['reference_text'], case['flac_sha256']) != (
                original['id'], original['speaker_id'], original['chapter_id'], original['samples'], original['text'], original['audio_sha256']):
            raise ValueError('Recording or human transcript identity differs')
        if case['language'] != 'en' or case['max_new_tokens'] != 444:
            raise ValueError('Transcription policy differs')


def validate_audio(path):
    audio = json.loads(path.read_text(encoding='utf-8'))
    pins_path = Path(__file__).with_name('dataset.json')
    pins = json.loads(pins_path.read_text(encoding='utf-8'))
    selection_path = path.parent / 'selection.json'
    if sha(pins_path) != audio['pins_sha256'] or sha(selection_path) != audio['selection_sha256']:
        raise ValueError('Selection or dataset pin bytes changed')
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    validate_selection(audio, selection, pins)
    for case in audio['cases']:
        for kind in ('pcm', 'features', 'flac'):
            candidate = Path(case[kind])
            if candidate.name != str(candidate) or sha(path.parent / candidate) != case[kind + '_sha256']:
                raise ValueError('Recording payload identity differs')
        pcm = np.load(path.parent / case['pcm'], allow_pickle=False)
        features = np.load(path.parent / case['features'], allow_pickle=False)
        if pcm.dtype != np.float32 or pcm.shape != (case['samples'],) or not np.isfinite(pcm).all() or not np.any(pcm):
            raise ValueError('PCM contract')
        if features.dtype != np.float32 or features.shape != (1, 128, 3000) or not np.isfinite(features).all():
            raise ValueError('Whisper feature contract')
    return audio


def validate_coverage(rows, names, repeated):
    if [r['name'] for r in rows] != names + (names[:1] if repeated else []):
        raise ValueError('Result coverage/order differs')
    if repeated and [r['repeat'] for r in rows] != [False] * len(names) + [True]:
        raise ValueError('Repeated request marker differs')


def decision(family, result, native=False):
    if family == 'parakeet':
        names = dict(text='Text', token_ids='TokenIds', frame_indices='FrameIndices', duration_frames='DurationFrames',
                     stop_reason='StopReason', encoded_frames='EncodedFrames', decoder_calls='DecoderCalls')
    else:
        names = dict(text='Text', token_ids='TokenIds', stop_reason='StopReason', skipped_as_no_speech='SkippedAsNoSpeech')
    if native and family == 'whisper':
        return dict(text=result['text'], token_ids=result['tokens'], stop_reason=result['stop_reason'], skipped_as_no_speech=result['skipped_as_no_speech'])
    return {key: result[key if native else source] for key, source in names.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'whisper-native', 'whisper-managed', 'parakeet-native', 'parakeet-managed', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Completed score exists')
    if (importlib.metadata.version('jiwer'), importlib.metadata.version('rapidfuzz')) != ('4.0.0', '3.14.6'):
        raise ValueError('Scoring dependency versions differ')
    audio = validate_audio(args.audio)
    names = [c['name'] for c in audio['cases']]
    models = {}
    core_ids, data_ids = set(), set()
    for family in ('parakeet', 'whisper'):
        native_path, managed_path = getattr(args, family + '_native'), getattr(args, family + '_managed')
        native = json.loads(native_path.read_text(encoding='utf-8'))
        managed = json.loads(managed_path.read_text(encoding='utf-8'))
        if native['audio_manifest_sha256'] != sha(args.audio) or managed['audio_manifest_sha256'] != sha(args.audio) or managed['native_manifest_sha256'] != sha(native_path):
            raise ValueError('Result provenance differs')
        if managed['schema'] != 1 or managed['family'] != family or managed['flags'] or (native['numpy'], native['onnxruntime']) != ('2.2.4', '1.29.0'):
            raise ValueError('Runtime policy differs')
        asset_path = Path(__file__).parents[2] / ('parakeet/transcribe/assets.json' if family == 'parakeet' else 'whisper/transcription-assets.json')
        if native['assets'] != json.loads(asset_path.read_text(encoding='utf-8')):
            raise ValueError('Native model pins differ')
        core_ids.add(managed['core_sha256']); data_ids.add(managed['data_sha256'])
        validate_coverage(native['cases'], names, family == 'parakeet')
        validate_coverage(managed['cases'], names, True)
        if decision(family, managed['cases'][0]['result']) != decision(family, managed['cases'][-1]['result']):
            raise ValueError('Managed repeated request differs')
        if family == 'parakeet' and native['cases'][0]['expected'] != native['cases'][-1]['expected']:
            raise ValueError('Native repeated request differs')
        rows = []
        for case, expected, actual in zip(audio['cases'], native['cases'][:20], managed['cases'][:20], strict=True):
            if expected['pcm_sha256'] != case['pcm_sha256'] or actual['pcm_sha256'] != case['pcm_sha256'] or not actual['input_and_held_results_unchanged']:
                raise ValueError('Result recording/ownership differs')
            left = decision(family, expected['expected'] if family == 'parakeet' else expected, True)
            right = decision(family, actual['result'])
            matches = left == right
            if matches != actual['matches']:
                raise ValueError('Reported application comparison is inconsistent')
            if family == 'parakeet' and (not expected['upstream_crosscheck'] or len(expected['steps']) != left['decoder_calls']):
                raise ValueError('Native upstream/step coverage differs')
            rows.append(dict(name=case['name'], speaker=case['speaker_id'], seconds=case['samples'] / 16000,
                human_text=case['reference_text'], native=left, managed=right, application_matches=matches,
                native_metrics=metrics(case['reference_text'], left['text']), managed_metrics=metrics(case['reference_text'], right['text'])))
        matches = all(row['application_matches'] for row in rows)
        repeat = managed['cases'][-1]
        if repeat['matches'] != (decision(family, repeat['result']) == decision(family, native['cases'][0]['expected'] if family == 'parakeet' else native['cases'][0], True)):
            raise ValueError('Repeated comparison inconsistent')
        if managed['passed'] != (matches and repeat['matches']) or not repeat['input_and_held_results_unchanged']:
            raise ValueError('Overall application/ownership claim inconsistent')
        totals = {}
        for engine in ('native', 'managed'):
            fields = ('reference_words', 'reference_characters', 'word_errors', 'character_errors', 'substitutions', 'deletions', 'insertions')
            total = {field: sum(row[engine + '_metrics'][field] for row in rows) for field in fields}
            total.update(word_error_rate=total['word_errors'] / total['reference_words'],
                         character_error_rate=total['character_errors'] / total['reference_characters'],
                         normalized_exact=sum(row[engine + '_metrics']['normalized_equal'] for row in rows))
            totals[engine] = total
        models[family] = dict(application_matches=matches, totals=totals, cases=rows,
            native_manifest_sha256=sha(native_path), managed_result_sha256=sha(managed_path),
            truncated={engine: [r['name'] for r in rows if r[engine]['stop_reason'] == 'TokenLimit'] for engine in ('native', 'managed')})
    if len(core_ids) != 1 or len(data_ids) != 1:
        raise ValueError('Managed model families used different product binaries')
    result = dict(schema=1, scope=audio['scope'], audio_manifest_sha256=sha(args.audio), scorer_sha256=sha(Path(__file__)),
        normalization='Unicode NFKC; uppercase; letters/numbers/internal ASCII apostrophes; other characters to space; collapse whitespace. CER includes normalized spaces.',
        dataset=audio['dataset'], core_sha256=next(iter(core_ids)), data_sha256=next(iter(data_ids)),
        utterances=20, speakers=10, seconds=sum(c['samples'] for c in audio['cases']) / 16000, models=models)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for family, model in models.items():
        print(family, 'native agreement', model['application_matches'], json.dumps(model['totals']))
    return 0 if all(v['application_matches'] for v in models.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
