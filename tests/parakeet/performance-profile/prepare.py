"""Build the diagnostic consumer against frozen production assemblies."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-performance-profile-20260921'
PRODUCT = ROOT/'artifacts/whisper-memory-product-v2-20260921/source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
MANIFEST = ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/parakeet.json'
SITE = ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def save(path, value):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8'); temporary.replace(path)


def verify(files):
    for name, wanted in files.items(): assert pin(ROOT/name) == wanted, name


def main():
    receipt_path = ROOT/'artifacts/asr-labeled-20260919/receipt.json'
    assert pin(receipt_path)['sha256'] == 'a4855ce825f4d5f1ea931440d031eb3a0aa5b3a0fee1fddbcf4d5b51500754b7'
    prior = read(receipt_path); spec = read(MANIFEST)
    native_path = ROOT/spec['reference']['path']; assert pin(native_path) == prior['files']['native-parakeet/manifest.json']
    native = read(native_path); assert native['onnxruntime'] == '1.29.0'
    assert len(spec['cases']) == 20 and sum(c['samples'] for c in spec['cases']) == 3412240
    for c, n in zip(spec['cases'], native['cases'][:20], strict=True):
        assert c['name'] == n['name'] and c['samples'] == n['samples'] and c['expected'] == n['expected']
        assert c['pcm']['sha256'] == n['pcm_sha256']
    files = {p.relative_to(ROOT).as_posix(): pin(p) for p in (MANIFEST, receipt_path, native_path)}
    for value in [*spec['models'].values(), *[c['pcm'] for c in spec['cases']]]:
        path = ROOT/value['path']; actual = pin(path); assert actual == {k: value[k] for k in ('bytes', 'sha256')}
        files[value['path']] = actual
    assert pin(PRODUCT/'Lokad.Onnx.dll')['sha256'] == 'd1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4'
    assert pin(PRODUCT/'Lokad.Onnx.Data.dll')['sha256'] == 'e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'
    BASE.mkdir(); source = BASE/'source'; source.mkdir(); (BASE/'logs').mkdir()
    for name in ('Program.cs', 'Profile.csproj'): shutil.copy2(TOOLS/name, source/name)
    shutil.copy2(ROOT/'tests/Shared/NpySupport.cs', source/'NpySupport.cs')
    flags = ['--tl:off', '--nologo', '-v', 'minimal', '-p:EnableSourceControlManagerQueries=false', '-p:EnableSourceLink=false', '-p:UseSharedCompilation=false', '-nr:false']
    commands = [ ['dotnet', 'restore', 'Profile.csproj', *flags, '--source', str(ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'), '--packages', str(BASE/'packages'), '-p:NuGetAudit=false'],
                 ['dotnet', 'build', 'Profile.csproj', '-c', 'Release', *flags, '--no-restore', '--disable-build-servers', '-p:FrozenProductDirectory='+str(PRODUCT)] ]
    builds = []; env = {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    for label, command in zip(('restore', 'build'), commands, strict=True):
        with (BASE/'logs'/(label+'.log')).open('x') as log:
            code = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300).returncode
        builds.append(dict(name=label, command=command, code=code)); save(BASE/'builds.json', builds)
        assert code == 0, label
    shutil.copytree(source/'bin/Release/net10.0', BASE/'bin')
    for path in PRODUCT.glob('*.dll'): shutil.copy2(path, BASE/'bin'/path.name)
    for folder in (source, BASE/'bin'):
        for path in folder.iterdir():
            if path.is_file(): files[path.relative_to(ROOT).as_posix()] = pin(path)
    files['tests/Shared/NpySupport.cs'] = pin(ROOT/'tests/Shared/NpySupport.cs')
    for path in TOOLS.glob('*.cs*'): files[path.relative_to(ROOT).as_posix()] = pin(path)
    verify(files)
    save(BASE/'prepared.json', dict(passed=True, files=files, builds=pin(BASE/'builds.json'), cases=20, decoder_calls_per_pass=1200,
        graph_calls=2480, graph_output_arrays=9760, input_arrays=12160, public_controls=20,
        source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope='Consumer build and original native public-fixture provenance; no inference yet'))
    print(json.dumps(dict(passed=True, prepared=pin(BASE/'prepared.json'), runner=pin(BASE/'bin/Profile.dll'))))


if __name__ == '__main__': main()
