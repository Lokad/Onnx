"""Freeze the qualified Data-only comparison before any new inference."""
import json
import shutil
from common import *


def main():
    prerequisites = [
        (QUALIFIED / 'closed.json', 'fb7d8a4df2a7e6f51896486d1f575605f40936ebc9d3d8b373aaf2840495d10a'),
        (PREDECESSOR / 'closed.json', 'c2e1d5da4746a6fb4cd4fe582663d922e074cb6bb4e03234757274cdfcf11a26')]
    for path, sha in prerequisites:
        assert pin(path)['sha256'] == sha
        closure = read(path)
        assert closure['passed']
        verify(closure['files'])
        for identity in closure['identities']:
            terminal(identity)
    assert read(prerequisites[0][0])['allocation_reduced']
    previous_path = OLD / 'timing/05-pyannote-ort/result.json'
    previous = read(previous_path)
    assert previous['manifest_sha256'] == pin(INPUT)['sha256']
    assert previous['runner_sha256'] == pin(NATIVE)['sha256']
    assert previous['adapter_sha256'] == pin(NATIVE.with_name('native_adapters.py'))['sha256']
    manifest = manifest_with_raw_hashes()
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    files = {}
    roles = {}
    for role in ['predecessor', 'candidate']:
        target = BASE / ('runtime-' + role)
        shutil.copytree(PREDECESSOR / 'bin', target)
        if role == 'candidate':
            shutil.copy2(QUALIFIED / 'application-runtime/Lokad.Onnx.Data.dll', target / 'Lokad.Onnx.Data.dll')
        roles[role] = {name: pin(target / name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll', 'AudioBenchmark.dll']}
        assert roles[role]['Lokad.Onnx.dll']['sha256'] == '469cb2d6a4558d917266434bd1f968c8b9f2762795b963800b2d945083852edd'
        assert roles[role]['AudioBenchmark.dll']['sha256'] == '7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
        for p in target.iterdir():
            if p.is_file():
                files[p.relative_to(ROOT).as_posix()] = pin(p)
    assert roles['predecessor']['Lokad.Onnx.Data.dll']['sha256'] == 'e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'
    assert roles['candidate']['Lokad.Onnx.Data.dll']['sha256'] == '1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662'
    for p in (BASE / 'runtime-predecessor').iterdir():
        if p.is_file() and p.name != 'Lokad.Onnx.Data.dll':
            assert pin(p) == pin(BASE / 'runtime-candidate' / p.name)
    assets = [*manifest['models'].values(), *manifest['native_assets'].values(), *manifest['upstream'].values(),
              manifest['reference'], *[c['pcm'] for c in manifest['cases']]]
    for asset in assets:
        assert pin(ROOT / asset['path']) == {k: asset[k] for k in ['bytes', 'sha256']}
        files[asset['path']] = pin(ROOT / asset['path'])
    external = {sys.executable: pin(Path(sys.executable))}
    for name, sha in previous['native_binaries'].items():
        assert pin(Path(name))['sha256'] == sha
        external[name] = pin(Path(name))
    paths = [*[p for p, sha in prerequisites], previous_path, INPUT, NATIVE, NATIVE.with_name('native_adapters.py'),
             MONITOR, ROOT / 'tests/audio/comparison/audit.py', ROOT / 'tests/pyannote/diarization/native_rules.py',
             QUALIFIED / 'dialogue-output/result.json', *TOOLS.iterdir()]
    for p in paths:
        if p.is_file():
            files[p.relative_to(ROOT).as_posix()] = pin(p)
    prepared = dict(passed=True, files=files, external_files=external, roles=roles,
        native_binaries=previous['native_binaries'], native_settings=previous['native_settings'],
        jobs=['predecessor', 'candidate', 'ort', 'ort', 'candidate', 'predecessor'],
        limits=dict(preflight_gib=10, rss_gib=8, seconds=1800),
        controls=dict(full_request_max_ratio=1.10, fixture_max_ratio=1.20),
        admission=dict(full_request_ratio=.97, maximum_fixture_ratio=1.05),
        scope='Descriptive Windows whole-application comparison; no calibrated parity or production promotion')
    save(BASE / 'prepared.json', prepared)
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'), jobs=6, calls=96)))


if __name__ == '__main__':
    main()
