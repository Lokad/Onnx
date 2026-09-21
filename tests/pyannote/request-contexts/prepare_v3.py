"""Replace the two optional arguments with explicit overloads; keep both failures."""
import difflib
import json
import shutil
import traceback
from common import *

TARGET = ROOT / 'artifacts/pyannote-request-contexts-v3-20260921'
SECOND = ROOT / 'artifacts/pyannote-request-contexts-v2-20260921'


def main():
    assert not TARGET.exists()
    receipt = SECOND / 'failure-closed.json'
    assert pin(receipt)['sha256'] == '8fde84f4d469ef392dae618c11530067458346df17dc80efed4d232fe73801b8'
    failure = read(receipt); assert not failure['passed']; verify(failure['files'])
    for identity in failure['identities']: terminal(identity)
    TARGET.mkdir(); (TARGET / 'logs').mkdir()
    source = TARGET / 'source'; shutil.copytree(BASE / 'source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    runtime = TARGET / 'runtime'; shutil.copytree(BASE / 'runtime', runtime)
    path = source / 'src/Lokad.Onnx.Data/WeSpeakerEmbedder.cs'; text = path.read_text(encoding='utf8')
    old = 'PipelineRequest? request = null'; assert text.count(old) == 1
    text = text.replace(old, 'PipelineRequest? request')
    old = '    // Contexts belong to one diarization embedding loop, never to the model instance.'; assert text.count(old) == 1
    text = text.replace(old, '''    internal WeSpeakerEmbedding[] ExtractPipeline(float[] samples, float[][] masks, CancellationToken cancellation) =>
        ExtractPipeline(samples, masks, cancellation, null);

'''+old)
    path.write_text(text, encoding='utf8')
    test = source / 'tests/Lokad.Onnx.Backend.Tests/PipelineRequestTests.cs'
    before = (SECOND / 'source/tests/Lokad.Onnx.Backend.Tests/PipelineRequestTests.cs').read_text(encoding='utf8')
    old = '    static WeSpeakerEmbedding[] Extract(object request, float[] samples, float[][] masks, CancellationToken token = default) =>'
    assert before.count(old) == 1
    after = before.replace(old, '''    static WeSpeakerEmbedding[] Extract(object request, float[] samples, float[][] masks) =>
        Extract(request, samples, masks, CancellationToken.None);
    static WeSpeakerEmbedding[] Extract(object request, float[] samples, float[][] masks, CancellationToken token) =>''')
    test.write_text(after, encoding='utf8')
    project = source / 'src/Lokad.Onnx.Data/Lokad.Onnx.Data.csproj'
    text = project.read_text(encoding='utf8'); assert text.count(str(BASE / 'runtime')) == 2
    project.write_text(text.replace(str(BASE / 'runtime'), str(runtime)), encoding='utf8')
    patch = ''
    for name in ['src/Lokad.Onnx.Data/WeSpeakerEmbedder.cs', 'src/Lokad.Onnx.Data/Community1Diarizer.cs']:
        old = (CANDIDATE / 'candidate-source' / name).read_text(encoding='utf8')
        new = (source / name).read_text(encoding='utf8')
        patch += ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='a/'+name, tofile='b/'+name))
    (TARGET / 'candidate.patch').write_text(patch, encoding='utf8')
    save(TARGET / 'source-changes.json', dict(failure=pin(receipt), test_diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True))),
        scope='Explicit overloads replace defaults; corrected nonlinear fixture retained; same request lifetime and all assertions.'))
    bridge = TARGET / 'bridge'; shutil.copytree(BASE / 'bridge', bridge, ignore=shutil.ignore_patterns('bin', 'obj'))
    path = bridge / 'Program.cs'; text = path.read_text(encoding='utf8')
    old = ': removed.Length == 1 && removed[0].StartsWith(pipeline) && added.Length == 5'; assert text.count(old) == 1
    text = text.replace(old, ': removed.Length == 0 && added.Length == 5')
    old = '&& differences.Length == 1 && differences[0].StartsWith("Lokad.Onnx.Community1Diarizer::Diarize::");'; assert text.count(old) == 1
    text = text.replace(old, '''&& differences.Length == 2
          && differences.Count(k => k.StartsWith("Lokad.Onnx.Community1Diarizer::Diarize::")) == 1
          && differences.Count(k => k.StartsWith(pipeline)) == 1;''')
    path.write_text(text, encoding='utf8')
    monitor.BASE = TARGET
    owner = psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    def command(name, args): return monitor.worker(state, TARGET / 'builds.json', name, args, source, [0], 8, 8, 900, True, None)
    try:
        for name, proj in [('data', project), ('bridge', bridge / 'Bridge.csproj')]:
            command(name+'-restore', ['dotnet', 'restore', proj, *flags, '--source', FEED, '--packages', TARGET / 'packages'])
            command(name+'-build', ['dotnet', 'build', proj, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
        shutil.copy2(source / 'src/Lokad.Onnx.Data/bin/Release/net10.0/Lokad.Onnx.Data.dll', runtime / 'Lokad.Onnx.Data.dll')
        assert pin(runtime / 'Lokad.Onnx.dll') == pin(CANDIDATE / 'runtimes/candidate/Lokad.Onnx.dll')
        command('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', CANDIDATE / 'runtimes/candidate', runtime, TARGET / 'instructions.json'])
        assert read(TARGET / 'instructions.json')['passed']; state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally: state['complete'] = True; save(TARGET / 'builds.json', state)
    files = {rel(receipt): pin(receipt), rel(MONITOR): pin(MONITOR), rel(BASE / 'prepared.json'): pin(BASE / 'prepared.json')}
    for folder in [TARGET, TOOLS]:
        for path in folder.rglob('*'):
            if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(folder).parts): files[rel(path)] = pin(path)
    save(TARGET / 'prepared.json', dict(passed=True, files=files, runtime={p.name:pin(p) for p in runtime.glob('*.dll')},
        scope='Explicit overload successor; Data-only request lifetime; public/model qualification pending.'))
    print(json.dumps(dict(passed=True, core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'), prepared=pin(TARGET / 'prepared.json'))))


if __name__ == '__main__': main()
