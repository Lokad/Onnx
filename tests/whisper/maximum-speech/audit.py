"""Audit maximum speech decisions, independent oracle, immutable payload and resources."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import sys
import numpy as np
from tokenizers import Tokenizer
from prepare import sha, read, write_new, pin


def validator_at(directory):
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('retained_recording_validator', directory/'recording_audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def application(managed, native, inputs, short, validator, tokenizer):
    assert managed['schema'] == native['schema'] == inputs['schema'] == 1
    assert inputs['sample_rate'] == 16000
    assert [c['name'] for c in inputs['cases']] == [c['name'] for c in native['cases']] == ['maximum-speech']
    case, reference = inputs['cases'][0], native['cases'][0]
    assert case['samples'] == 9600000 and case['language'] == 'en' and case['max_new_tokens'] == 444 and case['max_windows'] == 256
    assert managed['inputs_sha256'] == native['inputs_sha256']
    assert [c['name'] for c in managed['cases']] == ['maximum-speech']*2
    assert [c['repeat'] for c in managed['cases']] == [False, True]
    assert managed['refusals'] == 10 and managed['ownership'] is True and managed['flags'] == {}
    assert managed['cases'][0]['result'] == managed['cases'][1]['result'], 'Repeat changed result'
    validator.validate_recording(reference['result'], case, tokenizer)
    rows = []
    for actual in managed['cases']:
        assert actual['ownership'] is True
        assert actual['pcm_sha256'] == case['pcm_sha256'] == reference['pcm_sha256']
        assert math.isfinite(actual['seconds']) and actual['seconds'] > 0
        result = actual['result']
        validator.validate_recording(result, case, tokenizer)
        assert result['stop_reason'] == 'Completed' and result['processed_seconds'] == result['duration_seconds'] == 600
        assert len(result['windows']) >= 20 and result['segments'] and result['text'].strip()
        assert validator.decisions(result) == validator.decisions(reference['result']), 'Native application decisions differ'
        differences = []
        for x,y in zip(result['windows'], reference['result']['windows'], strict=True):
            for key in ('no_speech_probability', 'average_log_probability'):
                u,v = x['decoding'][key],y['decoding'][key]
                if u is None or v is None:
                    assert u is v
                else:
                    assert math.isfinite(u) and math.isfinite(v)
                    differences.append(abs(u-v))
        rows.append(dict(name=actual['name'], repeat=actual['repeat'], seconds=actual['seconds'],
            audio_seconds=600, windows=len(result['windows']), segments=len(result['segments']),
            tokens=sum(len(w['decoding']['token_ids']) for w in result['windows']),
            stop_reason=result['stop_reason'], maximum_native_confidence_difference=max(differences, default=0.)))
    assert managed['empty'] == dict(text='', segments=[], windows=[], stop_reason='Completed', duration_seconds=0, processed_seconds=0)
    assert len(managed['concurrent']) == 2 and managed['concurrent'][0] == managed['concurrent'][1]
    for result, samples, count in [(managed['silent'], 9600000, 20)]+[(r,496000,2) for r in managed['concurrent']]:
        validator.validate_recording(result, dict(samples=samples,max_new_tokens=444,max_windows=256), tokenizer)
        assert result['stop_reason'] == 'Completed' and result['text'] == '' and result['segments'] == []
        assert len(result['windows']) == count and all(w['decoding']['stop_reason'] == 'SilentInput' for w in result['windows'])
    actual,expected = managed['short_regression'],short['cases'][0]
    assert actual['text'] == expected['text'] and actual['token_ids'] == expected['tokens']
    assert actual['stop_reason'] == expected['stop_reason'] and actual['skipped_as_no_speech'] == expected['skipped_as_no_speech']
    ns,lp = actual['no_speech_probability'],actual['average_log_probability']
    assert math.isfinite(ns) and 0 <= ns <= 1 and math.isfinite(lp) and lp <= 0
    assert actual['skipped_as_no_speech'] == (ns>.6 and lp<=-1)
    return rows


def process(identity, samples, phase, frozen_sha):
    assert identity['schema'] == 1 and identity['phase'] == phase and identity['complete'] is True
    assert identity['code'] == 0 and 'error' not in identity and identity['frozen_sha256'] == frozen_sha
    assert identity['supervisor_affinity'] == [0]
    assert identity['limits'] == dict(rss=16*1024**3, seconds=7200, available_memory=1024**3)
    assert len(samples) == identity['samples'] and len(samples) > 1
    assert 0 < identity['seconds'] < 7200 and identity['ended'] >= identity['started']
    births, cpu = {}, {}
    peak, minimum, previous = 0, math.inf, -1
    for sample in samples:
        assert previous <= sample['seconds'] < identity['seconds']
        previous = sample['seconds']
        assert sample['rss'] == sum(m['rss'] for m in sample['members']) and 0 <= sample['rss'] < 16*1024**3
        assert sample['available_memory'] >= 1024**3
        peak, minimum = max(peak,sample['rss']), min(minimum,sample['available_memory'])
        seen = set()
        for member in sample['members']:
            pid,birth = member['pid'],member['create_time']
            assert pid not in seen and member['affinity'] == [2] and member['rss'] >= 0
            assert births.get(str(pid), birth) == birth and member['cpu_seconds'] >= cpu.get(pid,0)
            if pid == identity['pid']:
                assert birth == identity['create_time']
            seen.add(pid)
            births[str(pid)] = birth
            cpu[pid] = member['cpu_seconds']
    assert births == identity['members'] and births[str(identity['pid'])] == identity['create_time']
    assert identity['peak_rss'] == peak and identity['minimum_available_memory'] <= minimum
    return dict(phase=phase, seconds=identity['seconds'], samples=len(samples), peak_rss=peak,
                minimum_available_memory=minimum, terminal_processes=[dict(pid=int(pid),create_time=birth) for pid,birth in births.items()]+[
                    dict(pid=identity['supervisor'],create_time=identity['supervisor_create_time'])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    base = args.artifact.resolve()
    frozen = read(base/'frozen.json')
    for name, expected in frozen['files'].items():
        assert pin(base/name) == expected, name
    inputs = read(base/'inputs/inputs.json')
    native = read(base/'native/manifest.json')
    managed = read(base/'managed/result.json')
    assert native['inputs_sha256'] == managed['inputs_sha256'] == frozen['inputs_sha256'] == sha(base/'inputs/inputs.json')
    assert (native['numpy'],native['torch'],native['onnxruntime'],native['transformers']) == ('2.2.4','2.11.0+cpu','1.29.0','5.16.1')
    assert native['generator_sha256'] == sha(base/'native-source/recording/generate_reference.py')
    assert native['timestamp_helper_sha256'] == sha(base/'native-source/recording/generate_rules.py')
    assert native['source_files'] == inputs['sources']
    for name, expected in inputs['sources'].items():
        assert pin(base/'inputs/upstream'/Path(name).name) == expected
    assets = read(base/'native-source/transcription-assets.json')
    assert native['assets'] == assets
    models = Path(frozen['models'])
    for name, expected in assets['files'].items():
        assert pin(models/name) == expected
    for key, name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','RecordingReplay.dll')]:
        assert managed[key] == sha(base/'bin'/name)
    assert managed['core_sha256'] == '05884cfd524cc7130321f5dc1bcd0af17dddc7b97e8428d2d2f59e00edb795c2'
    assert managed['data_sha256'] == '27598aa8d8c6b97a1415302cf3aaced1adcf53b20b64734c0c0e047492ca069d'
    assert managed['runtime'] == '.NET 10.0.12' and 'Windows' in managed['os']
    assert managed['short_manifest_sha256'] == sha(base/'short/manifest.json')
    assert sha(base/'inputs/maximum-speech.npy') == inputs['cases'][0]['pcm_sha256']
    pcm = np.load(base/'inputs/maximum-speech.npy', allow_pickle=False, mmap_mode='r')
    assert pcm.dtype == np.float32 and pcm.shape == (9600000,) and np.isfinite(pcm).all()
    validator = validator_at(base/'reference')
    tokenizer = Tokenizer.from_file(str(models/'tokenizer.json'))
    rows = application(managed,native,inputs,read(base/'short/manifest.json'),validator,tokenizer)
    expected_arrays = set()
    steps = 0
    for i,window in enumerate(native['cases'][0]['result']['windows']):
        assert window['decoding']['stop_reason'] == 'EndToken'
        prefix = f'maximum-speech-w{i:03d}'
        expected_arrays.update([prefix+'-features.npy',prefix+'-hidden.npy'])
        tokens = window['decoding']['token_ids']
        steps += len(tokens)
        expected_arrays.update(prefix+f'-{step:03d}-logits.npy' for step in range(len(tokens)))
    assert set(native['files']) == expected_arrays == {p.name for p in (base/'native').glob('*.npy')}
    values = 0
    for name,item in native['files'].items():
        path = base/'native'/name
        assert path.name == name and pin(path) == {key:item[key] for key in ('bytes','sha256')}
        array = np.load(path,allow_pickle=False,mmap_mode='r')
        assert array.dtype == np.float32 and list(array.shape) == item['shape'] and np.isfinite(array).all()
        if name.endswith('-features.npy'):assert array.shape == (1,128,3000)
        elif name.endswith('-hidden.npy'):assert array.shape == (1,1500,1280)
        else:assert array.shape == (1,3 if name.endswith('-000-logits.npy') else 1,51866)
        values += array.size
    independent = read(base/'native-audit.json')
    assert independent['passed'] and independent['native_sha256'] == sha(base/'native/manifest.json')
    assert independent['inputs_sha256'] == sha(base/'inputs/inputs.json')
    assert independent['auditor_sha256'] == sha(base/'native-source/recording/audit_native.py')
    assert independent['timestamp_helper_sha256'] == native['timestamp_helper_sha256']
    assert independent['steps'] == steps and len(independent['windows']) == len(native['cases'][0]['result']['windows'])
    resources = []
    import psutil
    for phase in ('native','managed'):
        samples = [json.loads(line) for line in (base/(phase+'-samples.jsonl')).read_text(encoding='utf-8').splitlines()]
        resource = process(read(base/(phase+'-process.json')),samples,phase,sha(base/'frozen.json'))
        for identity in resource['terminal_processes']:
            try:
                assert psutil.Process(identity['pid']).create_time() != identity['create_time'], 'Owned process remains'
            except psutil.NoSuchProcess:
                pass
        resources.append(resource)
    result = dict(schema=1, passed=True, scope='Finite maximum-duration application and resource qualification; no natural long accuracy or full tensor claim',
        frozen_sha256=sha(base/'frozen.json'), native_sha256=sha(base/'native/manifest.json'), managed_sha256=sha(base/'managed/result.json'),
        native_audit_sha256=sha(base/'native-audit.json'), auditor_sha256=sha(Path(__file__)),
        native_arrays=len(expected_arrays), native_values=values, native_steps=steps, rows=rows, resources=resources,
        recording_calls=6, refusals=10, short_regression=True, ownership=True, process_peak_working_set=managed['peak_working_set'])
    write_new(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
