"""Audit Linux recording decisions and resources without rerunning inference."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import sys
from remote import h

NAMES = ['connected','shifted','token-limit','window-limit']


def load_validator(directory):
    directory = Path(directory)
    sys.path.insert(0,str(directory))
    spec = importlib.util.spec_from_file_location('retained_whisper_recording_audit',directory/'recording_audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def confidence_differences(actual,expected):
    assert len(actual['windows']) == len(expected['windows'])
    values = []
    for a,b in zip(actual['windows'],expected['windows'],strict=True):
        for key in ('no_speech_probability','average_log_probability'):
            x,y = a['decoding'][key],b['decoding'][key]
            if x is None or y is None:
                assert x is y
            else:
                assert type(x) in (int,float) and type(y) in (int,float) and math.isfinite(x) and math.isfinite(y)
                values.append(abs(x-y))
    return max(values,default=0.)


def application_audit(managed,native,windows,inputs,short,validator,tokenizer):
    assert managed['schema'] == native['schema'] == inputs['schema'] == 1
    assert [c['name'] for c in inputs['cases']] == [c['name'] for c in native['cases']] == NAMES
    assert native['onnxruntime'] == '1.29.0' and native['numpy'] == '2.2.4'
    assert inputs['sample_rate'] == 16000
    assert managed['ownership'] is True and managed['refusals'] == 10 and managed['flags'] == {}
    assert managed['inputs_sha256'] == native['inputs_sha256'] == windows['inputs_sha256']
    assert managed['short_manifest_sha256'] == windows['short_manifest_sha256']
    for group in (managed,windows):
        assert [c['name'] for c in group['cases']] == NAMES+NAMES[:1]
        assert [c['repeat'] for c in group['cases']] == [False]*4+[True]
    rows = []
    for i,actual in enumerate(managed['cases']):
        case,expected = inputs['cases'][i%4],native['cases'][i%4]
        assert actual['ownership'] is True and actual['pcm_sha256'] == case['pcm_sha256'] == expected['pcm_sha256']
        assert type(actual['seconds']) in (int,float) and math.isfinite(actual['seconds']) and actual['seconds'] > 0
        validator.validate_recording(actual['result'],case,tokenizer)
        validator.validate_recording(expected['result'],case,tokenizer)
        assert validator.decisions(actual['result']) == validator.decisions(expected['result']) == validator.decisions(windows['cases'][i]['result']), actual['name']
        rows.append(dict(name=actual['name'],repeat=actual['repeat'],seconds=actual['seconds'],
            audio_seconds=actual['result']['duration_seconds'],processed_seconds=actual['result']['processed_seconds'],
            windows=len(actual['result']['windows']),segments=len(actual['result']['segments']),
            tokens=sum(len(w['decoding']['token_ids']) for w in actual['result']['windows']),
            stop_reason=actual['result']['stop_reason'],maximum_native_confidence_difference=confidence_differences(actual['result'],expected['result']),
            maximum_windows_confidence_difference=confidence_differences(actual['result'],windows['cases'][i]['result'])))
    assert managed['cases'][0]['result'] == managed['cases'][-1]['result']
    assert managed['empty'] == dict(text='',segments=[],windows=[],stop_reason='Completed',duration_seconds=0,processed_seconds=0)
    assert len(managed['concurrent']) == 2 and managed['concurrent'][0] == managed['concurrent'][1]
    for name,result,samples in [('silent',managed['silent'],9600000),('concurrent',managed['concurrent'][0],496000)]:
        validator.validate_recording(result,dict(samples=samples,max_new_tokens=444,max_windows=256),tokenizer)
        assert validator.decisions(result) == validator.decisions(windows[name] if name == 'silent' else windows[name][0])
        assert result['text'] == '' and result['segments'] == [] and result['stop_reason'] == 'Completed'
        assert len(result['windows']) == (20 if name == 'silent' else 2)
        assert all(w['decoding']['stop_reason'] == 'SilentInput' for w in result['windows'])
    actual,expected = managed['short_regression'],short['cases'][0]
    assert type(actual['skipped_as_no_speech']) is bool and all(type(t) is int for t in actual['token_ids'])
    assert actual['text'] == expected['text'] and actual['token_ids'] == expected['tokens']
    assert actual['stop_reason'] == expected['stop_reason'] and actual['skipped_as_no_speech'] == expected['skipped_as_no_speech']
    assert validator.decisions(actual) == validator.decisions(windows['short_regression'])
    ns,lp = actual['no_speech_probability'],actual['average_log_probability']
    assert math.isfinite(ns) and 0 <= ns <= 1 and math.isfinite(lp) and lp <= 0
    assert actual['skipped_as_no_speech'] == (ns > .6 and lp <= -1)
    return rows


def process_audit(identity,sample_sets,collection):
    assert identity['schema'] == 1 and identity['complete'] and 'error' not in identity
    assert identity['supervisor']['affinity'] == '0'
    assert identity['limits'] == dict(rss=27*1024**3//2,seconds=1800,available_memory=256*1024**2)
    assert [r['name'] for r in identity['runs']] == ['managed','cli-connected']
    assert set(sample_sets) == {'managed','cli-connected'}
    terminal = {(identity['supervisor']['pid'],identity['supervisor']['start'])}
    previous_end = identity['started']
    resources = []
    for row in identity['runs']:
        assert row['code'] == 0 and 0 < row['seconds'] < 1800
        assert previous_end <= row['started'] < row['ended'] <= identity['ended']
        previous_end = row['ended']
        samples = sample_sets[row['name']]
        assert samples and len(samples) == row['samples']
        peak,minimum,previous_time = 0,math.inf,-1
        births,cpu = {},{}
        for sample in samples:
            assert previous_time < sample['seconds'] < min(row['seconds'],1800)
            previous_time = sample['seconds']
            assert sample['available_memory'] >= 256*1024**2
            minimum = min(minimum,sample['available_memory'])
            current = set()
            for member in sample['members']:
                pid,start = member['pid'],member['start']
                assert member['group'] == row['pid'] and member['affinity'] == '2' and start >= row['start']
                assert member['state'] != 'Z' and member['rss'] >= 0 and member['cpu_seconds'] >= cpu.get((pid,start),0)
                assert pid not in current and births.get(str(pid),start) == start
                if pid == row['pid']:assert start == row['start']
                current.add(pid)
                births[str(pid)] = start
                cpu[(pid,start)] = member['cpu_seconds']
            rss = sum(m['rss'] for m in sample['members'])
            assert rss < 27*1024**3//2
            peak = max(peak,rss)
        assert births == row['members'] and births[str(row['pid'])] == row['start']
        assert peak == row['peak_rss']
        terminal.update((int(pid),start) for pid,start in births.items())
        resources.append(dict(name=row['name'],seconds=row['seconds'],samples=len(samples),peak_rss=peak,minimum_available=minimum))
    assert terminal == {(p['pid'],p['start']) for p in collection['terminal_processes']}
    assert collection['complete'] and collection['code'] == 0
    assert collection['checkout'] == '172181fc5ab4eb2bdc2eb7f37e80d25e482a0887'
    return resources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    assert not a.output.exists()
    from tokenizers import Tokenizer
    import numpy as np
    root = Path(__file__).resolve().parents[3]
    base = a.artifact.resolve()
    payload,collected = base/'payload',base/'collected'
    bundle = h.read(payload/'bundle.json')
    assert h.sha(payload/'bundle.json') == h.sha(collected/'bundle.json') == h.read(base/'preparation.json')['bundle_sha256']
    collection = h.read(collected/'collection.json')
    assert h.sha(collected/'collection.json') == h.read(base/'download.json')['collection_sha256']
    assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()} == set(collection['files'])|{'collection.json'}
    for name,pin in collection['files'].items():h.verify_file(h.safe_path(collected,name),pin)
    for name,pin in bundle['files'].items():
        h.verify_file(h.safe_path(payload,name),pin)
        if name in collection['files']:h.verify_file(collected/name,pin)
    receipt = h.read(payload/'reference/windows-receipt.json')
    assert h.sha(payload/'reference/windows-receipt.json') == bundle['original_receipt_sha256'] == '623d1ad90efa120d0910f1f9fad92deaf3d4d9ad8410b74e8c10f16742d66275'
    for name,original in [('reference/windows-frozen.json','frozen.json'),('reference/native.json','native-corrected/manifest.json'),
                          ('reference/windows-managed.json','managed/result.json'),('inputs/inputs.json','inputs/inputs.json')]:
        assert h.sha(payload/name) == receipt['files'][original]
    frozen = h.read(payload/'reference/windows-frozen.json')
    for folder in ('recording','cli'):
        for filename,digest in frozen['binaries'][folder].items():
            if not filename.endswith('.exe'):assert h.sha(payload/'bin'/filename) == digest
    assert h.sha(payload/'reference/recording_audit.py') == receipt['audit_sources']['audit.py']
    assert h.sha(payload/'reference/score.py') == frozen['source']['tests/audio/accuracy/score.py']
    native = h.read(payload/'reference/native.json')
    windows = h.read(payload/'reference/windows-managed.json')
    managed = h.read(collected/'result/managed/result.json')
    inputs = h.read(payload/'inputs/inputs.json')
    short = h.read(payload/'short/manifest.json')
    assert managed['inputs_sha256'] == h.sha(payload/'inputs/inputs.json')
    assert managed['short_manifest_sha256'] == h.sha(payload/'short/manifest.json')
    for key,filename in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','RecordingReplay.dll')]:
        assert managed[key] == windows[key] == h.sha(payload/'bin'/filename)
    assert managed['runtime'] == '.NET 10.0.8' and 'Ubuntu' in managed['os']
    assets = h.read(payload/'reference/assets.json')
    assert native['assets'] == assets
    tokenizer_path = root/'models/whisper-large-v3-turbo/tokenizer.json'
    h.verify_file(tokenizer_path,assets['files']['tokenizer.json'])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for directory,cases in [(payload/'inputs',inputs['cases']),(payload/'short',short['cases'][:1])]:
        for case in cases:
            path = directory/case['pcm']
            assert h.sha(path) == case['pcm_sha256']
            pcm = np.load(path,allow_pickle=False)
            assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
    assert h.sha(payload/'inputs/connected.wav') == inputs['cases'][0]['wave_sha256']
    validator = load_validator(payload/'reference')
    rows = application_audit(managed,native,windows,inputs,short,validator,tokenizer)
    cli = h.read(collected/'result/cli-connected.stdout')
    validator.validate_recording(cli,inputs['cases'][0],tokenizer)
    assert validator.decisions(cli) == validator.decisions(managed['cases'][0]['result'])
    assert (collected/'result/cli-connected.stderr').read_text() == ''
    identity = h.read(collected/'result/identity.json')
    deployment = h.read(collected/'deployment.json')
    assert identity['bundle_sha256'] == h.sha(payload/'bundle.json') == deployment['bundle_sha256']
    assert (deployment['pid'],deployment['start']) == (identity['supervisor']['pid'],identity['supervisor']['start'])
    assert int((collected/'complete.txt').read_text()) == 0
    samples = {r['name']:[json.loads(line) for line in (collected/('result/'+r['name']+'-samples.jsonl')).read_text().splitlines()] for r in identity['runs']}
    resources = process_audit(identity,samples,collection)
    result = dict(schema=1,passed=True,recording_requests=9,main_requests=5,concurrent_silent_requests=2,cli_requests=1,refusals=10,short_regressions=1,
        fresh_native_inference=False,observations=rows,resources=resources,load_seconds=managed['load_seconds'],process_peak_working_set=managed['peak_working_set'],
        maximum_cli_api_confidence_difference=confidence_differences(cli,managed['cases'][0]['result']),runtime=managed['runtime'],os=managed['os'],
        core_sha256=managed['core_sha256'],data_sha256=managed['data_sha256'],runner_sha256=managed['runner_sha256'],
        bundle_sha256=h.sha(payload/'bundle.json'),collection_sha256=h.sha(collected/'collection.json'),audit_sha256=h.sha(__file__))
    h.write_new(a.output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
