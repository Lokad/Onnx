"""Audit preserved AMD recording outputs; no inference and no tensor-tolerance changes."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
from remote import read, sha, verify_file, write_new

NAMES = ['connected', 'shifted', 'hard-boundary', 'token-limit', 'window-limit', 'maximum-speech', 'tiny-tail', 'maximum-silence', 'empty']
REFUSALS = ['sample-rate', 'too-long', 'windows-0', 'windows-513', 'tokens-0', 'tokens-4097', 'per-frame-0', 'per-frame-11',
    'nonfinite-NaN', 'nonfinite-Infinity', 'nonfinite--Infinity', 'null-options', 'null-decoding', 'short-api-bound', 'pre-canceled', 'during-inference']


def process_audit(identity, sample_sets, collection):
    assert identity['schema'] == 1 and identity['complete'] and 'error' not in identity
    assert identity['supervisor']['affinity'] == '0'
    assert identity['limits'] == dict(rss=13*1024**3, seconds=1800, available_memory=256*1024**2)
    assert [r['name'] for r in identity['runs']] == ['managed', 'cli-connected', 'cli-token-limit']
    assert set(sample_sets) == {'managed', 'cli-connected', 'cli-token-limit'}
    expected_terminal = {(identity['supervisor']['pid'], identity['supervisor']['start'])}
    previous_end = identity['started']
    resources = []
    for row in identity['runs']:
        assert row['code'] == 0 and 0 < row['seconds'] < 1800
        assert previous_end <= row['started'] < row['ended'] <= identity['ended']
        previous_end = row['ended']
        samples = sample_sets[row['name']]
        assert len(samples) == row['samples'] and samples
        peak = 0
        minimum = math.inf
        prior = -1
        births = {}
        cpu = {}
        for sample in samples:
            assert prior < sample['seconds'] < min(row['seconds'], 1800)
            prior = sample['seconds']
            assert sample['available_memory'] >= 256*1024**2
            minimum = min(minimum, sample['available_memory'])
            current = set()
            for member in sample['members']:
                pid, start = member['pid'], member['start']
                assert member['group'] == row['pid'] and member['affinity'] == '2' and start >= row['start']
                assert member['state'] != 'Z' and member['rss'] >= 0 and member['cpu_seconds'] >= cpu.get((pid, start), 0)
                assert pid not in current and births.get(str(pid), start) == start
                if pid == row['pid']:
                    assert start == row['start']
                current.add(pid)
                births[str(pid)] = start
                cpu[(pid, start)] = member['cpu_seconds']
            rss = sum(m['rss'] for m in sample['members'])
            assert rss < 13*1024**3
            peak = max(peak, rss)
        assert births == row['members'] and births[str(row['pid'])] == row['start']
        assert peak == row['peak_rss']
        expected_terminal.update((int(pid), start) for pid, start in births.items())
        resources.append(dict(name=row['name'], seconds=row['seconds'], peak_rss=peak, minimum_available=minimum, samples=len(samples)))
    assert {(r['pid'],r['start']) for r in collection['terminal_processes']} == expected_terminal
    assert collection['complete'] and collection['code'] == 0
    assert collection['checkout'] == '172181fc5ab4eb2bdc2eb7f37e80d25e482a0887'
    return resources


def application_audit(managed, native, windows, inputs, validate, canonical_refusals, load_pcm, vocab):
    assert managed['schema'] == native['schema'] == inputs['schema'] == 1
    assert inputs['sample_rate'] == 16000 and [c['name'] for c in inputs['cases']] == NAMES
    assert native['ort'] == '1.29.0' and native['numpy'] == '2.2.4'
    assert managed['ownership'] is True and not managed['flags']
    assert canonical_refusals(managed['refusals']) == REFUSALS
    for group in (managed, native, windows):
        assert [r['name'] for r in group['cases']] == NAMES + NAMES[:1]
        assert [r['repeat'] for r in group['cases']] == [False]*9 + [True]
    observations = []
    checks = 0
    for i,(actual,expected,prior) in enumerate(zip(managed['cases'],native['cases'],windows['cases'],strict=True)):
        assert actual['result'] == expected['result'] == prior['result'], 'Application differs: '+actual['name']
        assert type(actual['seconds']) in (int,float) and math.isfinite(actual['seconds']) and actual['seconds'] > 0
        case = inputs['cases'][i%9]
        validate(actual['result'], case, load_pcm(case), vocab)
        checks += sum(w['decoding']['stop_reason'] != 'SilentInput' for w in actual['result']['windows'])
        observations.append(dict(name=actual['name'], repeat=actual['repeat'], seconds=actual['seconds'],
            audio_seconds=actual['result']['duration_seconds'], processed_seconds=actual['result']['processed_seconds'],
            windows=len(actual['result']['windows']), tokens=sum(len(w['decoding']['token_ids']) for w in actual['result']['windows']),
            stop_reason=actual['result']['stop_reason']))
    assert checks == native['upstream_crosschecks'] == 38
    assert len(managed['concurrent']) == 2 and managed['concurrent'] == windows['concurrent']
    assert managed['cases'][0]['result'] == managed['cases'][-1]['result']
    return observations


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    import numpy as np
    root = Path(__file__).resolve().parents[3]
    base = a.artifact.resolve()
    old = root/'artifacts/parakeet-recording-20260919'
    payload, collected = base/'payload', base/'collected'
    bundle = read(payload/'bundle.json')
    assert sha(payload/'bundle.json') == sha(collected/'bundle.json') == read(base/'preparation.json')['bundle_sha256']
    collection = read(collected/'collection.json')
    assert sha(collected/'collection.json') == read(base/'download.json')['collection_sha256']
    assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()} == set(collection['files']) | {'collection.json'}
    for name,pin in collection['files'].items():
        verify_file(collected/name, pin)
    for name,pin in bundle['files'].items():
        verify_file(old/name if name in bundle['recipes'] else payload/name, pin)
        if name in collection['files']:
            verify_file(collected/name, pin)
    receipt = read(payload/'reference/windows-receipt.json')
    assert receipt['closed'] and sha(payload/'reference/windows-receipt.json') == bundle['original_receipt_sha256'] == '2f23617863b6718677578b27357528dc1f8ece5e84cd0112a8cd7293839a65e6'
    for name,sources in bundle['original_paths'].items():
        if name.startswith(('bin/', 'inputs/')) or name.startswith('reference/') and name != 'reference/windows-receipt.json':
            for original in sources:
                assert receipt['files'][original] == bundle['files'][name]
    native = read(payload/'reference/native.json')
    windows = read(payload/'reference/windows-managed.json')
    managed = read(collected/'result/managed/result.json')
    inputs = read(payload/'inputs/inputs.json')
    assert managed['inputs_sha256'] == native['inputs_sha256'] == sha(payload/'inputs/inputs.json')
    for key,filename in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','ParakeetRecordingReplay.dll')]:
        assert managed[key] == windows[key] == bundle['files']['bin/'+filename]['sha256']
    assert managed['runtime'] == '.NET 10.0.8' and 'Ubuntu' in managed['os']
    assets = read(payload/'reference/assets.json')
    assert native['assets'] == assets
    models = root/'models/parakeet-tdt-0.6b-v3'
    # Remote installer and replay verify all model bytes; recheck local vocabulary for independent decoding.
    verify_file(models/'vocab.txt', assets['files']['vocab.txt'])
    vocab = {}
    for line in (models/'vocab.txt').read_text(encoding='utf-8').splitlines():
        piece,index = line.rsplit(' ',1)
        vocab[int(index)] = piece.replace('\u2581',' ')
    path = payload/'reference/recording-audit.py'
    spec = importlib.util.spec_from_file_location('retained_recording_validator', path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    def pcm(case):
        name = 'inputs/'+case['pcm']
        path = old/name if name in bundle['recipes'] else payload/name
        data = np.load(path, allow_pickle=False)
        assert data.dtype == np.float32 and data.shape == (case['samples'],) and np.isfinite(data).all()
        return data
    observations = application_audit(managed,native,windows,inputs,validator.validate,validator.canonical_refusals,pcm,vocab)
    validator.validate(managed['concurrent'][0],dict(max_tokens=4096,max_tokens_per_frame=10,max_windows=256),np.zeros(496000,np.float32),vocab)
    for name in ('connected','token-limit'):
        assert read(collected/('result/cli-'+name+'.stdout')) == next(r['result'] for r in native['cases'] if r['name'] == name)
        error = (collected/('result/cli-'+name+'.stderr')).read_text(encoding='utf-8')
        assert ('stopped before completion' in error) if name == 'token-limit' else error == ''
    identity = read(collected/'result/identity.json')
    assert identity['bundle_sha256'] == sha(payload/'bundle.json')
    deployment = read(collected/'deployment.json')
    assert (deployment['pid'], deployment['start']) == (identity['supervisor']['pid'], identity['supervisor']['start'])
    assert int((collected/'complete.txt').read_text()) == 0
    samples = {r['name']:[json.loads(line) for line in (collected/('result/'+r['name']+'-samples.jsonl')).read_text().splitlines()] for r in identity['runs']}
    resources = process_audit(identity,samples,collection)
    output = dict(schema=1, passed=True, recording_requests=10, concurrent_silent_requests=2, cli_requests=2, refusals=16,
        retained_native_window_crosschecks=38, fresh_native_inference=False, observations=observations, resources=resources,
        load_seconds=managed['load_seconds'], process_peak_working_set=managed['peak_working_set'], runtime=managed['runtime'],os=managed['os'],
        core_sha256=managed['core_sha256'],data_sha256=managed['data_sha256'],runner_sha256=managed['runner_sha256'],
        bundle_sha256=sha(payload/'bundle.json'),collection_sha256=sha(collected/'collection.json'),audit_sha256=sha(__file__))
    write_new(a.output,output)
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
