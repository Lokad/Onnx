"""Qualify sparse mel Data with frozen Core and original complete consumers."""
import difflib
import json
import re
import shutil
import traceback
import xml.etree.ElementTree as ET
from common import *


def main():
    assert not (BASE / 'qualification.json').exists()
    prepared = prerequisites()
    source = BASE / 'suites-source'; shutil.copytree(MODEL / 'source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    correct_cli(source)
    runtime = MODEL / 'runtime'
    dependencies = ['Lokad.Onnx', 'Lokad.Onnx.Data', 'Google.Protobuf', 'FastBertTokenizer', 'Lokad.Tokenizers', 'SixLabors.ImageSharp']
    projects = {name: source / path for name, path in [
        ('cli', 'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
        ('backend', 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
        ('tensors', 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')]}
    changes = []
    for name, path in projects.items():
        before = path.read_text(encoding='utf8')
        after, count = re.subn(r'    <ProjectReference Include="[^"]+" />', '', before)
        if name == 'backend':
            assert count == 0
            for n in dependencies:
                assert '<HintPath>'+str(runtime / (n+'.dll'))+'</HintPath>' in before
            changes.append(dict(path=path.relative_to(source).as_posix(), reason='Already references exact candidate runtime', diff=''))
            continue
        assert count == (1 if name == 'tensors' else 2)
        references = '\n'.join(f'    <Reference Include="{n}"><HintPath>{runtime / (n + ".dll")}</HintPath></Reference>' for n in dependencies)
        after = after.replace('</Project>', '  <ItemGroup>\n' + references + '\n  </ItemGroup>\n</Project>')
        path.write_text(after, encoding='utf8')
        changes.append(dict(path=path.relative_to(source).as_posix(), reason='Reference exact qualified Core/candidate Data; no product rebuild',
            diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))))
    save(BASE / 'suite-harness-changes.json', changes)
    own = psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'qualification.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    suites = []
    def command(name, args, cwd, *, is_test=False, inference=False, seconds=900, output=None):
        result = monitor.worker(state, BASE / 'qualification.json', name, args, cwd, [0], 10 if is_test or inference else 8,
            8, seconds, not inference, output)
        print(name, 'passed', flush=True)
        return result
    def test(name, project, filter=None):
        args = ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
            '--logger', 'trx;LogFileName='+name+'.trx', '--results-directory', BASE / 'test-results']
        if filter: args += ['--filter', filter]
        command(name, args, source, is_test=True, output=BASE / 'test-results')
        xml = ET.parse(BASE / 'test-results' / (name+'.trx')); counters = xml.find('.//{*}Counters').attrib
        assert int(counters['failed']) == 0 and int(counters['passed']) >= (2 if filter else 300)
        suites.append(dict(name=name, counters=counters)); save(BASE / 'suites.json', suites)
    try:
        for name, project in projects.items():
            command(name+'-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], source)
            command(name+'-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], source)
            for dependency in dependencies:
                assert pin(project.parent / 'bin/Release/net10.0' / (dependency+'.dll')) == pin(runtime / (dependency+'.dll'))
        reference = projects['backend'].parent / 'bin/Release/net10.0/dense-reference'
        reference.mkdir()
        shutil.copy2(PRIOR / 'application-runtime/Lokad.Onnx.Data.dll', reference / 'Lokad.Onnx.Data.dll')
        assert pin(reference / 'Lokad.Onnx.Data.dll')['sha256'] == '1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662'
        suite_files = dict(prepared['files'])
        for path in source.rglob('*'):
            if path.is_file() and 'obj' not in path.relative_to(source).parts: suite_files[rel(path)] = pin(path)
        suite_files[rel(Path(__file__))] = pin(Path(__file__))
        save(BASE / 'suites-prepared.json', dict(passed=True, files=suite_files, limits=dict(preflight_gib=10, rss_gib=8, seconds=900)))
        test('request-focused', projects['backend'], 'FullyQualifiedName~PipelineRequestTests')
        test('backend-full', projects['backend'])
        test('tensors-full', projects['tensors'])
        verify(suite_files)

        files = dict(suite_files)
        for parent, sha in [(PUBLIC, 'c2e1d5da4746a6fb4cd4fe582663d922e074cb6bb4e03234757274cdfcf11a26'),
            (MEETINGS, 'f9978dd22cec3091e640030e1db8b8e70aeb3c111c43d704c11f679ca5a89c1c')]:
            assert pin(parent / 'closed.json')['sha256'] == sha
            prior = read(parent / 'closed.json'); assert prior['passed']; verify(prior['files'])
            files.update(prior['files']); files[rel(parent / 'closed.json')] = pin(parent / 'closed.json')
        app_runtime = BASE / 'application-runtime'; shutil.copytree(runtime, app_runtime)
        for parent, stem in [(PUBLIC, 'AudioBenchmark'), (MEETINGS, 'NaturalMeetings')]:
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
                shutil.copy2(parent / 'bin' / (stem+'.'+suffix), app_runtime / (stem+'.'+suffix))
        assert pin(app_runtime / 'AudioBenchmark.dll')['sha256'] == '7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
        assert pin(app_runtime / 'NaturalMeetings.dll')['sha256'] == '79e3e7990ba6aa29e42da788277aad41b774ff3b8c3966b18ab1101944d0c0f1'
        meetings = BASE / 'meetings'; meetings.mkdir(); shutil.copytree(MEETINGS / 'inputs', meetings / 'inputs')
        manifest = read(MEETINGS / 'manifest.json')
        manifest.update(core_sha256=CORE, data_sha256=pin(runtime / 'Lokad.Onnx.Data.dll')['sha256'],
            accuracy_scope='Sparse mel weights; byte-identical accepted Core and original meeting consumer; no new native timing.')
        save(meetings / 'manifest.json', manifest)
        for folder in [app_runtime, meetings, TOOLS]:
            for path in folder.rglob('*'):
                if path.is_file(): files[rel(path)] = pin(path)
        for path in [INPUT, ROOT / 'tests/audio/comparison/audit.py', ROOT / 'tests/pyannote/natural-meetings/audit.py',
            ROOT / 'tests/pyannote/natural-meetings/common.py']:
            files[rel(path)] = pin(path)
        save(BASE / 'applications-prepared.json', dict(passed=True, files=files, public_input=rel(INPUT),
            limits=dict(preflight_gib=10, rss_gib=8, available_gib=1, disk_gib=20, output_gib=1, short_seconds=900, meeting_seconds=3600),
            scope='Original public consumers, output/allocation qualification only; no matched timing verdict.'))
        command('dialogue', ['dotnet', app_runtime / 'AudioBenchmark.dll', ROOT, INPUT, BASE / 'dialogue-output', 'timing'],
            ROOT, inference=True, output=BASE / 'dialogue-output')
        check = read(BASE / 'dialogue-output/result.json'); assert len(check['records']) == 16 and check['held_outputs_unchanged']
        old = {r['name']: r['result'] for r in read(PUBLIC / 'process/0-candidate/output/result.json')['records']}
        assert all(r['result'] == old[r['name']] for r in check['records']), 'Public decisions changed'
        for mode in ['inputs', 'run']:
            verify(files)
            command('meetings-'+mode, ['dotnet', app_runtime / 'NaturalMeetings.dll', ROOT, meetings, BASE / ('meetings-'+mode+'-output'), mode],
                ROOT, inference=True, seconds=3600, output=BASE / ('meetings-'+mode+'-output'))
        verify(files)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'qualification.json', state)


if __name__ == '__main__': main()
