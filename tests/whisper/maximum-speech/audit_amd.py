"""Independently check collected AMD maximum speech and cross-host decisions."""
from pathlib import Path
import argparse
import json
import math
from tokenizers import Tokenizer
from prepare import sha, read, write_new, pin
from audit import application, validator_at


def resources(identity, samples, collection):
    assert identity['schema'] == 1 and identity['complete'] and 'error' not in identity
    assert identity['supervisor']['affinity'] == '0'
    assert identity['limits'] == dict(rss=27*1024**3//2,seconds=3600,available_memory=256*1024**2)
    assert len(identity['runs']) == 1
    row = identity['runs'][0]
    assert row['name'] == 'managed' and row['code'] == 0 and 0 < row['seconds'] < 3600
    assert identity['started'] <= row['started'] <= row['ended'] <= identity['ended']
    assert len(samples) == row['samples'] and len(samples) > 1
    previous,peak,minimum = -1,0,math.inf
    births,cpu = {},{}
    for sample in samples:
        assert previous <= sample['seconds'] < row['seconds']
        previous = sample['seconds']
        assert sample['available_memory'] >= 256*1024**2
        minimum = min(minimum,sample['available_memory'])
        seen = set()
        for member in sample['members']:
            pid,start = member['pid'],member['start']
            assert member['group'] == row['pid'] and member['affinity'] == '2' and member['state'] != 'Z'
            assert start >= row['start'] and births.get(str(pid),start) == start
            assert member['rss'] >= 0 and member['cpu_seconds'] >= cpu.get(pid,0) and pid not in seen
            if pid == row['pid']:
                assert start == row['start']
            seen.add(pid)
            births[str(pid)] = start
            cpu[pid] = member['cpu_seconds']
        rss = sum(m['rss'] for m in sample['members'])
        assert rss < 27*1024**3//2
        peak = max(peak,rss)
    assert births == row['members'] and births[str(row['pid'])] == row['start'] and peak == row['peak_rss']
    expected = {(int(pid),start) for pid,start in births.items()}|{(identity['supervisor']['pid'],identity['supervisor']['start'])}
    assert expected == {(p['pid'],p['start']) for p in collection['terminal_processes']}
    assert collection['complete'] and collection['code'] == 0
    assert collection['checkout'] == '172181fc5ab4eb2bdc2eb7f37e80d25e482a0887'
    return dict(seconds=row['seconds'],samples=len(samples),peak_rss=peak,minimum_available_memory=minimum,
        terminal_processes=[dict(pid=pid,start=start) for pid,start in sorted(expected)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--windows',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    base,windows = args.artifact.resolve(),args.windows.resolve()
    payload,collected = base/'payload',base/'collected'
    bundle,collection = read(payload/'bundle.json'),read(collected/'collection.json')
    assert sha(payload/'bundle.json') == sha(collected/'bundle.json') == read(base/'preparation.json')['bundle_sha256']
    assert sha(collected/'collection.json') == read(base/'download.json')['collection_sha256']
    assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()} == set(collection['files'])|{'collection.json'}
    for name,expected in collection['files'].items():
        assert pin(collected/name) == expected,name
    for name,expected in bundle['files'].items():
        assert pin(payload/name) == expected,name
        if name in collection['files']:
            assert pin(collected/name) == expected
    frozen = read(windows/'frozen.json')
    assert sha(windows/'frozen.json') == sha(payload/'reference/windows-frozen.json') == bundle['windows_frozen_sha256']
    for name,expected in frozen['files'].items():
        assert pin(windows/name) == expected,name
    waudit = read(windows/'audit.json')
    assert waudit['passed'] and waudit['frozen_sha256'] == sha(windows/'frozen.json')
    native = read(payload/'reference/native.json')
    managed = read(collected/'result/managed/result.json')
    prior = read(windows/'managed/result.json')
    inputs = read(payload/'inputs/inputs.json')
    assert sha(payload/'reference/native.json') == waudit['native_sha256'] == sha(windows/'native/manifest.json')
    assert sha(windows/'managed/result.json') == waudit['managed_sha256']
    assert sha(payload/'reference/native-audit.json') == waudit['native_audit_sha256'] == sha(windows/'native-audit.json')
    assert managed['inputs_sha256'] == sha(payload/'inputs/inputs.json') == sha(windows/'inputs/inputs.json')
    assert managed['short_manifest_sha256'] == sha(payload/'short/manifest.json') == sha(windows/'short/manifest.json')
    for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','RecordingReplay.dll')]:
        assert managed[key] == prior[key] == sha(payload/'bin'/name) == sha(windows/'bin'/name)
    assert managed['runtime'] == '.NET 10.0.8' and 'Ubuntu' in managed['os']
    assets = read(payload/'reference/assets.json')
    assert assets == native['assets']
    model = Path(frozen['models'])
    assert pin(model/'tokenizer.json') == assets['files']['tokenizer.json']
    assert sha(payload/'inputs/maximum-speech.npy') == inputs['cases'][0]['pcm_sha256']
    validator = validator_at(payload/'reference')
    tokenizer = Tokenizer.from_file(str(model/'tokenizer.json'))
    rows = application(managed,native,inputs,read(payload/'short/manifest.json'),validator,tokenizer)
    for row,actual,expected in zip(rows,managed['cases'],prior['cases'],strict=True):
        assert validator.decisions(actual['result']) == validator.decisions(expected['result'])
        differences = [abs(a['decoding'][key]-b['decoding'][key]) for a,b in zip(actual['result']['windows'],expected['result']['windows'],strict=True)
            for key in ('no_speech_probability','average_log_probability') if a['decoding'][key] is not None]
        row['maximum_windows_confidence_difference'] = max(differences,default=0.)
    for key in ('empty','silent','concurrent','short_regression'):
        assert validator.decisions(managed[key]) == validator.decisions(prior[key])
    identity = read(collected/'result/identity.json')
    assert identity['bundle_sha256'] == sha(payload/'bundle.json')
    sample_set = [json.loads(line) for line in (collected/'result/managed-samples.jsonl').read_text(encoding='utf-8').splitlines()]
    resource = resources(identity,sample_set,collection)
    result = dict(schema=1,passed=True,rows=rows,resources=resource,windows_audit_sha256=sha(windows/'audit.json'),
        collection_sha256=sha(collected/'collection.json'),bundle_sha256=sha(payload/'bundle.json'),auditor_sha256=sha(Path(__file__)),
        managed_sha256=sha(collected/'result/managed/result.json'),recording_calls=6,refusals=10,short_regression=True,
        process_peak_working_set=managed['peak_working_set'],scope='Finite600-second cyclic speech API/resource and cross-host application proof; no full tensor/independent natural accuracy claim')
    write_new(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
