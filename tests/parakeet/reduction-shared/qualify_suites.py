"""Run repository suites against the exact model-qualified Core/Data bytes."""
import difflib
import json
import re
import shutil
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reduction-dispatch'))
from common import *

SUITES = ROOT / 'artifacts/parakeet-reduction-dispatch-suites-20260921'
SHARED = ROOT / 'artifacts/parakeet-reduction-shared-v2-20260921'


def main():
    closure = read(BASE / 'closed.json')
    assert closure['evidence_passed'] and closure['native_numeric_passed']
    assert pin(BASE / 'closed.json')['sha256'] == '0f5700924712ad313a840e28f704d22bb3f1c97c948264d5abd5769f31dfcaf1'
    verify(closure['files'])
    for identity in closure['terminal_identities']:
        terminal(identity)
    assert pin(SHARED / 'closed.json')['sha256'] == '85c1137baf975881e36b30ba72298ee343f26f22af0062b4a6ef7e6de1341d0a'
    shared = read(SHARED / 'closed.json')
    assert shared['passed']
    verify(shared['files'])
    for identity in shared['terminal_identities']: terminal(identity)
    hashes = read(SHARED / 'analysis.json')['dino_hashes']
    constant = {r['name']: r for r in hashes if r['scenario'] == 'constant'}
    pair = (constant['last_hidden_state']['hash'], constant['pooler_output']['hash'])
    assert pair == (12398120018957570767, 204586709688817035)
    assert all(r['maximum'] <= 1e-4 for r in constant.values())
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
    # This additional frozen pair is accepted only after complete native-array qualification above.
    dino = source / 'tests/Lokad.Onnx.Backend.Tests/GraphExecutionDinoV3Tests.cs'
    before = dino.read_text(encoding='utf8')
    marker = '            (12756423648221837382UL, 6147948217399512682UL)'
    assert before.count(marker) == 1
    after = before.replace(marker, marker + ',\n            // Corrected tensor dispatch: every output independently checked against ORT1.29.\n            (12398120018957570767UL, 204586709688817035UL)')
    dino.write_text(after, encoding='utf8')
    changes.append(dict(path=dino.relative_to(source).as_posix(), reason='Add complete native-qualified partial-sum hash pair; retain both original pairs and all numeric checks',
                        shared_closure=pin(SHARED / 'closed.json'),
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
         scope='Exact model-qualified binaries; three project references, known archived style helper and native-qualified Dino hash pair differ'))
    monitor.BASE = SUITES
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    feed = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
    results = []
    def command(label, args):
        is_test = label.endswith('-tests')
        return monitor.worker(state, SUITES / 'processes.json', label, args, source,
            [0, 1] if is_test else [0], 10 if is_test else 4, 8 if is_test else 2,
            900, True, SUITES / 'test-results')
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
            assert int(counters['passed']) > 300
            failures = [dict(name=r.attrib['testName'], message=r.find('.//{*}Message').text)
                        for r in xml.findall('.//{*}UnitTestResult') if r.attrib['outcome'] == 'Failed']
            assert len(failures) == int(counters['failed'])
            assert state['runs'][-1]['code'] == (0 if not failures else 1)
            results.append(dict(suite=label, counters=counters, failures=failures))
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
        assert run['complete'] and run['code'] in run['expected'] and run['samples'] > 0
        for pid, birth in run['members'].items():
            identity = dict(pid=int(pid), birth=birth)
            terminal(identity)
            identities.append(identity)
        samples = [json.loads(s) for s in (SUITES / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] and max(s['rss'] for s in samples) == run['peak_rss']
        is_test = run['name'].endswith('-tests')
        assert run['preflight']['available'] >= (10 if is_test else 4) * 1024**3
        assert all(s['rss'] < (8 if is_test else 2) * 1024**3 and s['seconds'] < 900 and s['available'] >= 1024**3 and s['disk'] >= 20 * 1024**3
                   and s['output_bytes'] <= 1024**3 and all(m['affinity'] == [2] for m in s['members']) for s in samples)
    files = {rel(p): pin(p) for p in SUITES.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(SUITES).parts)}
    for p in (Path(__file__), BASE / 'closed.json', SHARED / 'closed.json'):
        files[rel(p)] = pin(p)
    save(SUITES / 'closed.json', dict(passed=all(not r['failures'] for r in results), evidence_passed=True, files=files, results=results, worker_identities=identities,
         supervisor=state['supervisor'], resource_samples=sum(r['samples'] for r in state['runs']),
         peak_rss=max(r['peak_rss'] for r in state['runs']), model_closure=pin(BASE / 'closed.json')))
    print(json.dumps(dict(passed=all(not r['failures'] for r in results), results=results, closed=pin(SUITES / 'closed.json'))))


if __name__ == '__main__':
    main()
