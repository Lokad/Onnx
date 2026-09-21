"""Qualify optional LSTM panel admission in a fresh local source tree."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-lstm-panel-admission-20260921'
BASELINE_PROOF = None
ROWS = ROOT/'artifacts/pyannote-conv-row-sharing-v2-20260921'
PAYLOAD = ROOT/'artifacts/pyannote-amd-candidates-v2-20260921/payload'
HELPER = '''
        internal static int StorageLength(int inputWeights, int recurrentWeights)
        {
            if (inputWeights < 0 || recurrentWeights < 0) return 0;
            long count = (long)inputWeights + recurrentWeights;
            return count <= Array.MaxLength ? (int)count : 0;
        }
'''


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def main():
    sys.path.insert(0, str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    assert pin(ROWS/'prepared.json')['sha256'] == '724630645d35446464ce298c182b3bd7f2ff79d9e6d01e50a4ebe540bde09f1a'
    assert read(ROWS/'local-preparation.json')['local_preparation_passed'] is True
    for name, wanted in read(ROWS/'prepared.json')['files'].items(): assert pin(ROOT/name) == wanted, name
    spec = read(PAYLOAD/'payload.json')
    assert pin(PAYLOAD/'payload.json')['sha256'] == 'b5dee4ba28e64f62008d76fa763d51730ed9ccbc440335cd9688b0b6a2e233d0'
    for name, wanted in spec['nuget']['archives'].items(): assert pin(PAYLOAD/'nuget-feed'/name) == wanted
    if BASELINE_PROOF is not None:
        failure_path = BASELINE_PROOF/'failed-preparation.json'
        assert pin(failure_path)['sha256'] == '570ed87cfdb0aa0781ea9b89887fe1a8718b7495973a2dab1545816bea771466'
        failure = read(failure_path); assert failure['negative_control_passed'] and not failure['passed']
        for name, wanted in failure['files'].items(): assert pin(BASELINE_PROOF/name) == wanted, name
        assert pin(TOOLS/'LstmPanelOverflowRefusalTests.cs') == pin(BASELINE_PROOF/'failed-tools/LstmPanelOverflowRefusalTests.cs')
    BASE.mkdir(); (BASE/'logs').mkdir()
    for role in (('candidate',) if BASELINE_PROOF is not None else ('baseline', 'candidate')):
        source = BASE/(role+'-source')
        shutil.copytree(ROWS/'candidate-source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
        assert (source/'Lokad.Onnx.slnx').exists()
        shutil.copy2(TOOLS/'LstmPanelOverflowRefusalTests.cs', source/'tests/Lokad.Onnx.Backend.Tests/LstmPanelOverflowRefusalTests.cs')
    source = BASE/'candidate-source'; path = source/'src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs'
    before = path.read_text(); old = '            int count = checked(w.Length + r.Length);'
    assert before.count(old) == 1
    after = before.replace(old, '            int count = StorageLength(w.Length, r.Length);\n            if (count == 0) return null;')
    needle = '        public void Dispose() => ArrayPool<float>.Shared.Return(storage);\n'
    assert after.count(needle) == 1
    after = after.replace(needle, needle+HELPER); path.write_text(after, encoding='utf8')
    (BASE/'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs', tofile='b/src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs')), encoding='utf8')
    shutil.copy2(TOOLS/'LstmPanelAdmissionTests.cs', source/'tests/Lokad.Onnx.Backend.Tests/LstmPanelAdmissionTests.cs')
    # Compare all other compiled methods, retaining the exact new admission IL.
    bridge = BASE/'bridge'; bridge.mkdir()
    original_bridge = ROOT/'tests/whisper/memory-product-v2/CompareIlStable.cs'
    program = original_bridge.read_text()
    old = '''    if (oldMethods.Count == 0 || !oldMethods.Keys.Order().SequenceEqual(newMethods.Keys.Order())) throw new InvalidDataException("Method coverage differs");
    var differences = oldMethods.Keys.Where(k => oldMethods[k] != newMethods[k]).ToArray();
    if (differences.Length != 0) throw new InvalidDataException("Runtime IL differs: " + string.Join("; ", differences));'''
    new = '''    var added = newMethods.Keys.Except(oldMethods.Keys).Order().ToArray();
    if (oldMethods.Count == 0 || oldMethods.Keys.Except(newMethods.Keys).Any()) throw new InvalidDataException("Removed methods");
    var differences = oldMethods.Keys.Where(k => oldMethods[k] != newMethods[k]).Order().ToArray();
    string[] expectedAdded = name == "Lokad.Onnx.dll" ? ["Lokad.Onnx.CPUExecutionProvider+LstmProjectionPanels::StorageLength::Int32 StorageLength(Int32, Int32)"] : [];
    string[] expectedChanged = name == "Lokad.Onnx.dll" ? ["Lokad.Onnx.CPUExecutionProvider+LstmProjectionPanels::Create::LstmProjectionPanels Create(System.ReadOnlySpan`1[System.Single], System.ReadOnlySpan`1[System.Single], Int32, Int32, Int32, Int32, Lokad.Onnx.TensorExecutionOptions)"] : [];
    if (!added.SequenceEqual(expectedAdded) || !differences.SequenceEqual(expectedChanged)) throw new InvalidDataException("Unexpected method changes: " + string.Join("; ", added.Concat(differences)));'''
    assert program.count(old) == 1; program = program.replace(old, new)
    old = '        normalized_methods = oldMethods, equal = true });'
    new = '''        normalized_methods = oldMethods, unchanged_methods = oldMethods.Count - differences.Length,
        changed_methods = differences.ToDictionary(k => k, k => newMethods[k]),
        added_methods = added.ToDictionary(k => k, k => newMethods[k]), equal_except_admission = true });'''
    assert program.count(old) == 1; program = program.replace(old, new)
    program = program.replace('All Core/Data method IL, resolved operands, locals, stack and exception clauses',
                              'Only optional LSTM admission may change; all other Core/Data IL, operands, locals, stack and exception clauses match')
    (bridge/'Program.cs').write_text(program, encoding='utf8')
    (bridge/'Bridge.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup></Project>\n')
    own = psutil.Process(); prior_affinity = own.cpu_affinity(); own.cpu_affinity([0])
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE/'processes.json', state)
    env = {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    flags = ['--tl:off', '--nologo', '-v', 'minimal', '-p:EnableSourceControlManagerQueries=false',
             '-p:EnableSourceLink=false', '-p:UseSharedCompilation=false', '-nr:false']
    started = time.monotonic()
    def command(name, args, cwd, environment=env, expected=0):
        assert psutil.virtual_memory().available >= 8*1024**3
        row = dict(name=name, command=list(map(str, args)), expected=expected, complete=False, code=None,
                   samples=0, peak_rss=0, members={}); state['runs'].append(row); save(BASE/'processes.json', state)
        child = None; beginning = time.monotonic()
        try:
            with (BASE/'logs'/(name+'.log')).open('x') as log, (BASE/'logs'/(name+'.samples.jsonl')).open('x') as samples:
                own.cpu_affinity([2])
                try:
                    child = subprocess.Popen(row['command'], cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                finally:
                    own.cpu_affinity([0])
                process = psutil.Process(child.pid); row['child'] = dict(pid=child.pid, birth=process.create_time())
                row['members'][str(child.pid)] = row['child']['birth']; save(BASE/'processes.json', state)
                while child.poll() is None:
                    members = []
                    try:
                        for p in [process]+process.children(recursive=True):
                            try:
                                birth = p.create_time(); assert row['members'].get(str(p.pid), birth) == birth
                                row['members'][str(p.pid)] = birth
                                members.append(dict(pid=p.pid, birth=birth, rss=p.memory_info().rss, affinity=p.cpu_affinity()))
                            except psutil.NoSuchProcess: pass
                    except psutil.NoSuchProcess: pass
                    if not members and child.poll() is not None: break
                    sample = dict(seconds=time.monotonic()-beginning, available=psutil.virtual_memory().available, members=members)
                    samples.write(json.dumps(sample)+'\n'); samples.flush(); row['samples'] += 1
                    row['peak_rss'] = max(row['peak_rss'], sum(m['rss'] for m in members)); save(BASE/'processes.json', state)
                    assert sample['seconds'] < 900 and sample['available'] >= 1024**3 and sum(m['rss'] for m in members) < 8*1024**3
                    assert all(m['affinity'] == [2] for m in members)
                    assert time.monotonic()-started < 3600
                    time.sleep(.25)
                row['code'] = child.wait(); assert row['code'] == expected, (name, row['code'])
        except BaseException:
            for pid, birth in reversed(list(row['members'].items())):
                try:
                    p = psutil.Process(int(pid))
                    if p.create_time() == birth: p.kill()
                except psutil.NoSuchProcess: pass
            if child is not None: child.wait(timeout=15)
            raise
        finally:
            row.update(complete=True, seconds=time.monotonic()-beginning)
            if child is not None: row['code'] = child.poll()
            save(BASE/'processes.json', state)
        print(name, 'expected exit', expected, flush=True)
    def restore_build(label, project, cwd):
        command(label+'-restore', ['dotnet', 'restore', project, *flags, '--source', PAYLOAD/'nuget-feed',
                '--packages', BASE/'packages', '--no-http-cache', '-p:NuGetAudit=false'], cwd)
        command(label+'-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], cwd)
    def test(label, project, cwd, filtering=None, environment=env, expected=0):
        args = ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                '--logger', 'trx;LogFileName='+label+'.trx', '--results-directory', BASE/'test-results']
        if filtering: args += ['--filter', filtering]
        command(label, args, cwd, environment, expected)
    try:
        backend = Path('tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj')
        if BASELINE_PROOF is None:
            baseline = BASE/'baseline-source'
            restore_build('baseline', backend, baseline)
            test('baseline-refusal', backend, baseline, 'FullyQualifiedName~LstmPanelOverflowRefusalTests', expected=1)
            assert 'System.OverflowException' in (BASE/'logs/baseline-refusal.log').read_text(), 'Negative control must fail at the original checked addition'
        else:
            save(BASE/'negative-control-reuse.json', dict(failed_predecessor=pin(BASELINE_PROOF/'failed-preparation.json'),
                original_negative_control=pin(BASELINE_PROOF/'test-results/baseline-refusal.trx'), unchanged_test=pin(TOOLS/'LstmPanelOverflowRefusalTests.cs')))
        restore_build('candidate', backend, source)
        test('candidate-lstm', backend, source, 'FullyQualifiedName~Lstm')
        test('candidate-fallback', backend, source, 'FullyQualifiedName~LstmOutputLane|FullyQualifiedName~LstmPanel',
             dict(env, DOTNET_EnableHWIntrinsic='0'))
        restore_build('bridge', bridge/'Bridge.csproj', bridge)
        command('instruction-bridge', ['dotnet', bridge/'bin/Release/net10.0/Bridge.dll', ROWS/'runtime',
                source/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0', BASE/'instructions.json'], source)
        restore_build('cli', Path('src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'), source)
        tensors = Path('tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')
        restore_build('tensors', tensors, source)
        test('candidate-backend', backend, source)
        test('candidate-tensors', tensors, source)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state.update(complete=True, seconds=time.monotonic()-started); save(BASE/'processes.json', state); own.cpu_affinity(prior_affinity)
    original_sources = {p.relative_to(ROWS/'candidate-source').as_posix(): pin(p) for p in (ROWS/'candidate-source').rglob('*')
                        if p.is_file() and not {'bin', 'obj'}.intersection(p.relative_to(ROWS/'candidate-source').parts)}
    for name, wanted in original_sources.items():
        if name != 'src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs': assert pin(source/name) == wanted, name
    receipt = dict(passed=True, parent_core=pin(ROWS/'runtime/Lokad.Onnx.dll'),
                   candidate_core=pin(source/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0/Lokad.Onnx.dll'),
                   instructions=pin(BASE/'instructions.json'), source_files=original_sources, original_bridge=pin(original_bridge),
                   negative_control_predecessor=None if BASELINE_PROOF is None else pin(BASELINE_PROOF/'failed-preparation.json'),
                   tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()},
                   scope='Local optional-admission correctness only; no new model timing, AMD qualification or production promotion')
    save(BASE/'prepared.json', receipt); print(json.dumps({k: v for k, v in receipt.items() if k not in ('source_files', 'tools')}))


if __name__ == '__main__':
    main()
