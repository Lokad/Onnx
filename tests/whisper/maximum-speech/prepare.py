"""Prepare a new maximum-duration replay of the already qualified Whisper product."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def copy(source, destination, expected=None):
    if expected is not None:
        assert sha(source) == expected, str(source)
    assert not destination.exists(), str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert sha(source) == sha(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    old = root/'artifacts/whisper-recording-v2-20260919'
    parakeet = root/'artifacts/parakeet-recording-20260919'
    assert sha(old/'receipt.json') == '623d1ad90efa120d0910f1f9fad92deaf3d4d9ad8410b74e8c10f16742d66275'
    assert sha(parakeet/'receipt.json') == '2f23617863b6718677578b27357528dc1f8ece5e84cd0112a8cd7293839a65e6'
    receipt, previous = read(old/'receipt.json'), read(parakeet/'receipt.json')
    for name in ('frozen.json', 'reference-frozen.json', 'inputs/inputs.json', 'managed/result.json'):
        assert sha(old/name) == receipt['files'][name], name
    assert receipt['all_owned_workers_terminal'] and receipt['payload_files_unchanged']
    frozen, reference = read(old/'frozen.json'), read(old/'reference-frozen.json')
    for name in ('inputs/inputs.json', 'inputs/maximum-speech.npy', 'inputs/connected.npy'):
        assert pin(parakeet/name) == previous['files'][name], name
    data = read(old/'inputs/inputs.json')
    pcase = next(c for c in read(parakeet/'inputs/inputs.json')['cases'] if c['name'] == 'maximum-speech')
    pcm = np.load(parakeet/'inputs'/pcase['pcm'], allow_pickle=False)
    connected = np.load(parakeet/'inputs/connected.npy', allow_pickle=False)
    assert pcm.dtype == connected.dtype == np.float32 and pcm.shape == (9600000,)
    assert np.isfinite(pcm).all() and np.array_equal(pcm.view(np.uint32), np.resize(connected, 9600000).view(np.uint32))
    base = args.artifact.resolve()
    base.mkdir(parents=True, exist_ok=False)
    for name in ('receipt.json', 'frozen.json', 'reference-frozen.json'):
        copy(old/name, base/'reference'/('previous-'+name))
    copy(parakeet/'receipt.json', base/'reference/parakeet-receipt.json')
    copy(parakeet/'inputs/inputs.json', base/'reference/parakeet-inputs.json')
    copy(parakeet/'inputs'/pcase['pcm'], base/'inputs/maximum-speech.npy', pcase['pcm_sha256'])
    for name, item in data['sources'].items():
        copy(old/'inputs/upstream'/Path(name).name, base/'inputs/upstream'/Path(name).name, item['sha256'])
    inputs = dict(schema=1, sample_rate=16000,
        scope='Constructed cyclic speech for finite maximum-duration API qualification; not natural long-conversation accuracy',
        source_revision=data['source_revision'], sources=data['sources'],
        predecessor_receipt_sha256=sha(parakeet/'receipt.json'), predecessor_inputs_sha256=sha(parakeet/'inputs/inputs.json'),
        recipe=dict(kind='repeat-and-truncate', connected_sha256=sha(parakeet/'inputs/connected.npy'), connected_samples=len(connected), samples=9600000),
        cases=[dict(name='maximum-speech', pcm='maximum-speech.npy', pcm_sha256=pcase['pcm_sha256'],
            samples=9600000, language='en', max_new_tokens=444, max_windows=256)])
    write_new(base/'inputs/inputs.json', inputs)
    for name in ('generate_reference.py', 'generate_rules.py', 'sources.json'):
        rel = 'tests/whisper/recording/'+name
        copy(old/'reference-source'/rel, base/'native-source/recording'/name, reference['files'][rel])
    copy(old/'reference-source/tests/whisper/transcription-assets.json', base/'native-source/transcription-assets.json',
         reference['files']['tests/whisper/transcription-assets.json'])
    copy(root/'tests/whisper/recording/audit_native.py', base/'native-source/recording/audit_native.py')
    copy(root/'tests/whisper/recording/audit.py', base/'reference/recording_audit.py', receipt['audit_sources']['audit.py'])
    copy(old/'source/tests/audio/accuracy/score.py', base/'reference/score.py', frozen['source']['tests/audio/accuracy/score.py'])
    original = old/'source/tests/whisper/recording/Program.cs'
    copy(original, base/'reference/OriginalProgram.cs', frozen['source']['tests/whisper/recording/Program.cs'])
    text = original.read_text(encoding='utf-8')
    before = 'new[]{"connected","shifted","token-limit","window-limit"}'
    assert text.count(before) == 1
    text = text.replace(before, 'new[]{"maximum-speech"}')
    before = 'string serialized=JsonSerializer.Serialize(result,json);'
    assert text.count(before) == 1
    text = text.replace(before, 'Require(result.StopReason==WhisperRecordingStopReason.Completed && result.ProcessedSeconds==600 && result.DurationSeconds==600 && result.Windows.Count>=20,"Maximum speech did not complete");\n    '+before)
    build = base/'build'
    build.mkdir()
    (build/'Program.cs').write_text(text, encoding='utf-8')
    copy(old/'source/tests/Shared/NpySupport.cs', build/'NpySupport.cs', frozen['source']['tests/Shared/NpySupport.cs'])
    copy(base/'native-source/transcription-assets.json', build/'assets.json')
    (build/'RecordingReplay.csproj').write_text('''<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings><IsPackable>false</IsPackable></PropertyGroup>
  <ItemGroup>
    <Reference Include="Lokad.Onnx"><HintPath>../bin/Lokad.Onnx.dll</HintPath></Reference>
    <Reference Include="Lokad.Onnx.Data"><HintPath>../bin/Lokad.Onnx.Data.dll</HintPath></Reference>
    <EmbeddedResource Include="assets.json" LogicalName="assets.json" />
  </ItemGroup>
</Project>
''', encoding='utf-8')
    for name, digest in frozen['binaries']['recording'].items():
        if not name.endswith(('.exe', '.pdb')) and name != 'RecordingReplay.dll':
            copy(old/'recording-bin'/name, base/'bin'/name, digest)
    command = ['dotnet', 'build', str(build/'RecordingReplay.csproj'), '--tl:off', '--nologo', '-v', 'minimal', '-c', 'Release', '-o', str(base/'bin')]
    with (base/'build.log').open('x', encoding='utf-8') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (base/'build.log').read_text(encoding='utf-8')
    assert '0 Warning(s)' in log and '0 Error(s)' in log, log
    for name in ('RecordingReplay.deps.json', 'RecordingReplay.runtimeconfig.json'):
        assert sha(old/'recording-bin'/name) == frozen['binaries']['recording'][name]
        shutil.copyfile(old/'recording-bin'/name, base/'bin'/name)
    for name, expected in [('Lokad.Onnx.dll', '05884cfd524cc7130321f5dc1bcd0af17dddc7b97e8428d2d2f59e00edb795c2'),
                           ('Lokad.Onnx.Data.dll', '27598aa8d8c6b97a1415302cf3aaced1adcf53b20b64734c0c0e047492ca069d')]:
        assert sha(base/'bin'/name) == expected
    short = root/'artifacts/asr-labeled-20260919/native-whisper/manifest.json'
    copy(short, base/'short/manifest.json', read(old/'managed/result.json')['short_manifest_sha256'])
    first = read(short)['cases'][0]
    copy(short.parent/first['pcm'], base/'short'/first['pcm'], first['pcm_sha256'])
    source_paths = list(Path(__file__).parent.glob('*.py'))+[root/'.agent/m4-whisper-maximum-speech-20260919.md', root/'eng/campaign_processes.py']
    for path in source_paths:
        copy(path, base/'source'/path.relative_to(root))
    files = {path.relative_to(base).as_posix():pin(path) for path in sorted(base.rglob('*'))
             if path.is_file() and '/obj/' not in path.as_posix()}
    write_new(base/'frozen.json', dict(schema=1, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        product_source=frozen['commit'], inputs_sha256=sha(base/'inputs/inputs.json'), files=files,
        previous_receipt_sha256=sha(old/'receipt.json'), models=str(root/'models/whisper-large-v3-turbo')))
    print(json.dumps(dict(files=len(files), runner_sha256=sha(base/'bin/RecordingReplay.dll'), frozen_sha256=sha(base/'frozen.json')), indent=2))


if __name__ == '__main__':
    main()
