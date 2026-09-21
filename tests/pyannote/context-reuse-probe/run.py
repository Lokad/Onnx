"""Run a prospective, bounded graph diagnostic without rebuilding the product."""
import json
import shutil
import traceback
from common import *


def main():
    assert not BASE.exists(), 'Existing artifact'
    qualification = CANDIDATE / 'qualification-closed.json'
    assert pin(qualification)['sha256'] == 'cf950ec5cedf702c1af38decc377cd516a5d8b652f77853d03d2c81db0b5bf53'
    qualified = read(qualification); assert qualified['passed']
    prior = PROFILE / 'closed.json'
    assert pin(prior)['sha256'] == '4b7542a8917e558c92ba632fada6830ddf8c870fccf49064dae17ff1de194d5c'
    closed = read(prior); assert closed['passed']; verify(closed['files'])
    BASE.mkdir(); (BASE / 'logs').mkdir(); (BASE / 'source').mkdir(); (BASE / 'runtime').mkdir()
    owner = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    try:
        for name in ['Program.cs', 'Probe.csproj']:
            shutil.copy2(TOOLS / name, BASE / 'source' / name)
        files = {rel(prior): pin(prior), rel(qualification): pin(qualification), rel(INPUT): pin(INPUT)}
        files.update(closed['files'])
        for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll', 'Google.Protobuf.dll', 'FastBertTokenizer.dll', 'Lokad.Tokenizers.dll', 'SixLabors.ImageSharp.dll']:
            path = CANDIDATE / 'runtimes/candidate' / name
            expected = qualified['files'].get(str(path.relative_to(ROOT)), qualified['files'].get(rel(path)))
            assert pin(path) == expected, name
            files[rel(path)] = expected
            shutil.copy2(path, BASE / 'runtime' / name)
        flags = ['-c', 'Release', *monitor.FLAGS, '-o', str(BASE / 'bin'), '-p:FrozenProductDirectory=' + str(BASE / 'runtime')]
        monitor.worker(state, BASE / 'processes.json', 'build', ['dotnet', 'build', str(BASE / 'source/Probe.csproj'), *flags],
            BASE / 'source', [0], 10, 8, 900, True, BASE / 'bin')
        for path in (BASE / 'runtime').glob('*.dll'):
            target = BASE / 'bin' / path.name
            if not target.exists(): shutil.copy2(path, target)
            assert pin(target) == pin(path), path.name
        original = read(INPUT)
        cases = []
        for row in read(PROFILE / 'output/result.json')['rows']:
            if row['phase'] != 'unprofiled': continue
            case = dict(name=row['name'], graph=row['model'],
                input_name='waveform' if row['model'] == 'segmentation' else 'fbank_features',
                output_name='scores' if row['model'] == 'segmentation' else '/resnet/pool/Reshape_output_0')
            for source, target in [('input', 'input'), ('output', 'expected')]:
                item = row[source]; path = PROFILE / 'output' / item['file']; identity = pin(path)
                assert identity['sha256'] == item['sha256'] and identity['bytes'] == item['values'] * 4
                case[target] = dict(path=rel(path), shape=item['shape'], **identity)
            cases.append(case)
        models = dict(segmentation=original['models']['segmentation'], embedding=original['models']['encoder'])
        for item in models.values():
            assert pin(ROOT / item['path']) == {k: item[k] for k in ['bytes', 'sha256']}
            files[item['path']] = pin(ROOT / item['path'])
        save(BASE / 'manifest.json', dict(models=models, cases=cases, modes=['fresh', 'reuse', 'no-cache'], passes=3))
        for folder in [TOOLS, BASE / 'runtime', BASE / 'source', BASE / 'bin']:
            for path in folder.iterdir():
                if path.is_file(): files[rel(path)] = pin(path)
        files[rel(MONITOR)] = pin(MONITOR); files[rel(BASE / 'manifest.json')] = pin(BASE / 'manifest.json')
        save(BASE / 'prepared.json', dict(passed=True, files=files, scope='Graph allocation/ownership diagnostic, no application timing claim.',
            admission='Both orders must reduce repeat-pass embedding allocation with unchanged bits and passing resources.',
            limits=dict(preflight_gib=10, rss_gib=8, seconds=900, available_gib=1, disk_gib=20, output_gib=1), orders=['forward', 'reverse']))
        for order in ['forward', 'reverse']:
            verify(files)
            monitor.worker(state, BASE / 'processes.json', order,
                ['dotnet', str(BASE / 'bin/Probe.dll'), str(ROOT), str(BASE / 'manifest.json'), str(BASE / order), order],
                ROOT, [0], 10, 8, 900, False, BASE / order)
            result = read(BASE / order / 'result.json'); assert result['passed'] and len(result['records']) == 54
            print(order, '54 graph contracts passed', flush=True)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'processes.json', state)


if __name__ == '__main__': main()
