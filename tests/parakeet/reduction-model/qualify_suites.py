"""Run repository suites against the exact model-qualified Core/Data bytes."""
import difflib
import json
import re
import shutil
import traceback
import xml.etree.ElementTree as ET
from common import *

SUITES = ROOT / 'artifacts/parakeet-reduction-suites-20260921'


def main():
    closure = read(BASE / 'closed.json')
    assert closure['evidence_passed'] and closure['native_numeric_passed']
    assert pin(BASE / 'closed.json')['sha256'] == '71f833ec9260a26d38456ad641a3a436ea215ac44f70b760aabaef826e128742'
    verify(closure['files'])
    for identity in closure['terminal_identities']:
        terminal(identity)
    SUITES.mkdir()
    (SUITES / 'logs').mkdir()
    source = SUITES / 'source'
    shutil.copytree(BASE / 'source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    runtime = BASE / 'runtime'
    dependencies = ['Lokad.Onnx', 'Lokad.Onnx.Data', 'Google.Protobuf', 'FastBertTokenizer', 'Lokad.Tokenizers', 'SixLabors.ImageSharp']
    projects = {'cli': source / 'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj',
                'backend': source / 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',
                'tensors': source / 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'}
    changes = []
    for project in projects.values():
        before = project.read_text(encoding='utf8')
        after, count = re.subn(r'    <ProjectReference Include="[^"]+" />', '', before)
        assert count == (1 if project == projects['tensors'] else 2)
        references = '\n'.join(f'    <Reference Include="{name}"><HintPath>{runtime / (name + ".dll")}</HintPath></Reference>' for name in dependencies)
        after = after.replace('</Project>', '  <ItemGroup>\n' + references + '\n  </ItemGroup>\n</Project>')
        project.write_text(after, encoding='utf8')
        changes.append(dict(path=project.relative_to(source).as_posix(), reason='Reference exact qualified binaries instead of rebuilding product',
                            diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))))
    # Known archived test-tool style defect; fix only the scanned helper copy.
    helper = source / 'tests/whisper/weight-sharing/WhisperDecoderWeightsTests.cs'
    before = helper.read_text(encoding='utf8')
    old = '''    static DenseTensor<float> Weight(float[]? values = null, int[]? shape = null) =>
        new DenseTensor<float>(values ?? Enumerable.Range(0, 1024).Select(i => (float)i).ToArray(), shape ?? new[] { 32, 32 });'''
    new = '''    static DenseTensor<float> Weight() => Weight(new[] { 32, 32 });
    static DenseTensor<float> Weight(int[] shape) => Weight(Enumerable.Range(0, 1024).Select(i => (float)i).ToArray(), shape);
    static DenseTensor<float> Weight(float[] values) => Weight(values, new[] { 32, 32 });
    static DenseTensor<float> Weight(float[] values, int[] shape) => new DenseTensor<float>(values, shape);'''
    assert before.count(old) == 1
    after = before.replace(old, new)
    helper.write_text(after, encoding='utf8')
    changes.append(dict(path=helper.relative_to(source).as_posix(), reason='Known archived optional-parameter style defect; explicit equivalent overloads',
                        diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))))
    save(SUITES / 'test-harness-changes.json', changes)
    allowed = {r['path'] for r in changes}
    source_pins = {}
    for p in (BASE / 'source').rglob('*'):
        name = p.relative_to(BASE / 'source')
        if not p.is_file() or {'bin', 'obj'}.intersection(name.parts):
            continue
        if name.as_posix() not in allowed:
            assert pin(p) == pin(source / name), str(name)
        source_pins[rel(source / name)] = pin(source / name)
    save(SUITES / 'prepared.json', dict(passed=True, model_closure=pin(BASE / 'closed.json'), source_files=source_pins,
         runtime={name: pin(runtime / (name + '.dll')) for name in dependencies}, tool=pin(Path(__file__)),
         scope='Exact model-qualified binaries; only three project references and one known archived style helper differ'))
    monitor.BASE = SUITES
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    feed = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
    results = []
    def command(label, args):
        return monitor.worker(state, SUITES / 'processes.json', label, args, source, [0], 4, 2, 900, True, SUITES / 'test-results')
    try:
        for label, project in projects.items():
            command(label + '-restore', ['dotnet', 'restore', project, *flags, '--source', feed, '--packages', SUITES / 'packages'])
            command(label + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
            for name in dependencies:
                assert pin(project.parent / 'bin/Release/net10.0' / (name + '.dll')) == pin(runtime / (name + '.dll'))
        for label in ('backend', 'tensors'):
            command(label + '-tests', ['dotnet', 'test', projects[label], '-c', 'Release', *flags, '--no-build', '--no-restore',
                    '--logger', 'trx;LogFileName=' + label + '.trx', '--results-directory', SUITES / 'test-results'])
            xml = ET.parse(SUITES / 'test-results' / (label + '.trx'))
            counters = xml.find('.//{*}Counters').attrib
            assert int(counters['failed']) == 0 and int(counters['passed']) > 300
            results.append(dict(suite=label, counters=counters))
        verify(source_pins)
        verify(closure['files'])
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(SUITES / 'processes.json', state)
    identities = []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['samples'] > 0
        for pid, birth in run['members'].items():
            identity = dict(pid=int(pid), birth=birth)
            terminal(identity)
            identities.append(identity)
        samples = [json.loads(s) for s in (SUITES / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] and max(s['rss'] for s in samples) == run['peak_rss']
        assert run['preflight']['available'] >= 4 * 1024**3
        assert all(s['rss'] < 2 * 1024**3 and s['seconds'] < 900 and s['available'] >= 1024**3 and s['disk'] >= 20 * 1024**3
                   and s['output_bytes'] <= 1024**3 and all(m['affinity'] == [2] for m in s['members']) for s in samples)
    files = {rel(p): pin(p) for p in SUITES.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(SUITES).parts)}
    for p in (Path(__file__), BASE / 'closed.json'):
        files[rel(p)] = pin(p)
    save(SUITES / 'closed.json', dict(passed=True, files=files, results=results, worker_identities=identities,
         supervisor=state['supervisor'], resource_samples=sum(r['samples'] for r in state['runs']),
         peak_rss=max(r['peak_rss'] for r in state['runs']), model_closure=pin(BASE / 'closed.json')))
    print(json.dumps(dict(passed=True, results=results, closed=pin(SUITES / 'closed.json'))))


if __name__ == '__main__':
    main()
