"""Build and qualify only the inclusive constant-packing boundary, offline."""
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
import zipfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-packing-admission-20260921'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
PREDECESSOR = None
sys.path.insert(0, str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def pin(path):
    with path.open('rb') as f:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def save(path, value):
    for attempt in range(20):
        try:
            tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
            tmp.replace(path); return
        except PermissionError:
            if attempt == 19: raise
            time.sleep(.05)


def main():
    if PREDECESSOR is not None:
        failure = json.loads((PREDECESSOR/'failed-preparation.json').read_text(encoding='utf8'))
        assert failure['passed'] is False
        for name,wanted in failure['files'].items():assert pin(PREDECESSOR/name)==wanted,name
    BASE.mkdir(); (BASE/'logs').mkdir()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'archive', '--format=zip', '--output='+str(BASE/'source.zip'), commit], cwd=ROOT, check=True)
    sources = {}
    with zipfile.ZipFile(BASE/'source.zip') as archive:
        for role in (('baseline', 'candidate') if PREDECESSOR is None else ('candidate',)):
            source = BASE/(role+'-source'); source.mkdir(); archive.extractall(source); sources[role] = source
            shutil.copy2(TOOLS/'PackedBoundaryTests.cs', source/'tests/Lokad.Onnx.Backend.Tests/PackedBoundaryTests.cs')
    if PREDECESSOR is not None:sources['baseline']=PREDECESSOR/'baseline-source'
    candidate = sources['candidate']; target = candidate/'src/Lokad.Onnx/GraphPacking.cs'
    original = target.read_text(encoding='utf8'); old = 'n < MaxPackedAxis'
    assert original.count(old) == 1
    changed = original.replace(old, 'n <= MaxPackedAxis').replace(
        'Upper axis bound of measured packed-kernel territory (P46: 4096, covering GPT-2 c_proj at n=3072; census shows no other model edge in range).',
        'Inclusive reduction-axis bound for prepared packed weights; includes Parakeet feed-forward projections at n=4096.').replace(
        'reduction axis below MaxPackedAxis with total bytes below MaxPackedBytes',
        'reduction axis at most MaxPackedAxis with total bytes at most MaxPackedBytes')
    target.write_text(changed, encoding='utf8')
    if PREDECESSOR is not None:
        test_path=candidate/'tests/Lokad.Onnx.Backend.Tests/FoldBudgetTests.cs'
        before=test_path.read_text(encoding='utf8');old='[InlineData(4096, 1, false)]'
        assert before.count(old)==1
        test_path.write_text(before.replace(old,'[InlineData(4096, 1, true)]\n    [InlineData(4097, 1, false)]'),encoding='utf8')
    (BASE/'candidate.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
        fromfile='a/src/Lokad.Onnx/GraphPacking.cs', tofile='b/src/Lokad.Onnx/GraphPacking.cs')), encoding='utf8')
    bridge = BASE/'bridge'; bridge.mkdir()
    bridge_source = ROOT/'tests/whisper/memory-product-v2/CompareIlStable.cs'
    program = bridge_source.read_text(encoding='utf8')
    old = '    if (differences.Length != 0) throw new InvalidDataException("Runtime IL differs: " + string.Join("; ", differences));'
    new = '''    string[] expected = name == "Lokad.Onnx.dll" ? ["Lokad.Onnx.GraphPacking::FitsPackBudget::Boolean FitsPackBudget(Int32, Int32)"] : [];
    if (!differences.SequenceEqual(expected)) throw new InvalidDataException("Unexpected IL changes: " + string.Join("; ", differences));'''
    assert program.count(old) == 1; program = program.replace(old, new)
    old = '        normalized_methods = oldMethods, equal = true });'
    assert program.count(old) == 1
    program = program.replace(old, '''        normalized_methods = oldMethods, unchanged_methods = oldMethods.Count - differences.Length,
        changed_methods = differences.ToDictionary(k => k, k => newMethods[k]), equal_except_admission = true });''')
    program = program.replace('All Core/Data method IL, resolved operands, locals, stack and exception clauses',
                              'Only FitsPackBudget changes; remaining Core/Data IL, operands, locals, stack and exception clauses match')
    (bridge/'Program.cs').write_text(program, encoding='utf8')
    (bridge/'Bridge.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup></Project>\n')
    own = psutil.Process(); prior_affinity = own.cpu_affinity(); own.cpu_affinity([0])
    state = dict(complete=False, code=None, source=commit, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    flags = ['--tl:off', '--nologo', '-v', 'minimal', '-p:EnableSourceControlManagerQueries=false',
             '-p:EnableSourceLink=false', '-p:UseSharedCompilation=false', '-nr:false']
    started = time.monotonic()

    def command(name, args, cwd, expected=0, environment=env):
        assert psutil.virtual_memory().available >= 8*1024**3
        row = dict(name=name, command=list(map(str,args)), expected=expected, complete=False, code=None, samples=0, peak_rss=0, members={})
        state['runs'].append(row); save(BASE/'processes.json', state)
        child = None; beginning = time.monotonic()
        try:
            with (BASE/'logs'/(name+'.log')).open('x', encoding='utf8') as log, (BASE/'logs'/(name+'.samples.jsonl')).open('x') as samples:
                own.cpu_affinity([2])
                try:
                    child = subprocess.Popen(row['command'], cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                finally: own.cpu_affinity([0])
                process = psutil.Process(child.pid); row['members'][str(child.pid)] = process.create_time()
                while child.poll() is None:
                    members = []
                    try:
                        for p in [process]+process.children(recursive=True):
                            try:
                                birth = p.create_time(); assert row['members'].get(str(p.pid),birth) == birth
                                row['members'][str(p.pid)] = birth
                                members.append(dict(pid=p.pid,birth=birth,rss=p.memory_info().rss,affinity=p.cpu_affinity()))
                            except psutil.NoSuchProcess: pass
                    except psutil.NoSuchProcess: pass
                    if not members and child.poll() is not None: break
                    sample = dict(seconds=time.monotonic()-beginning, available=psutil.virtual_memory().available, members=members)
                    samples.write(json.dumps(sample)+'\n'); samples.flush(); row['samples'] += 1
                    row['peak_rss'] = max(row['peak_rss'], sum(m['rss'] for m in members)); save(BASE/'processes.json',state)
                    assert sample['seconds'] < 900 and sample['available'] >= 1024**3 and sum(m['rss'] for m in members) < 8*1024**3
                    assert all(m['affinity'] == [2] for m in members) and time.monotonic()-started < 3600
                    time.sleep(.25)
                row['code'] = child.wait(); assert row['code'] == expected,(name,row['code'])
        except BaseException:
            for pid,birth in reversed(list(row['members'].items())):
                try:
                    p=psutil.Process(int(pid))
                    if p.create_time()==birth: p.kill()
                except psutil.NoSuchProcess: pass
            if child is not None: child.wait(timeout=15)
            raise
        finally:
            row.update(complete=True,seconds=time.monotonic()-beginning)
            if child is not None: row['code']=child.poll()
            save(BASE/'processes.json',state)
        print(name,'expected exit',expected,flush=True)

    def build(label, project, cwd):
        command(label+'-restore',['dotnet','restore',project,*flags,'--source',FEED,'--packages',BASE/'packages','-p:NuGetAudit=false'],cwd)
        command(label+'-build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],cwd)

    def test(label, project, cwd, filtering=None, expected=0, environment=env):
        args=['dotnet','test',project,'-c','Release',*flags,'--no-build','--no-restore','--logger','trx;LogFileName='+label+'.trx','--results-directory',BASE/'test-results']
        if filtering: args += ['--filter',filtering]
        command(label,args,cwd,expected,environment)

    try:
        backend=Path('tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj')
        if PREDECESSOR is None:
            build('baseline',backend,sources['baseline'])
            test('baseline-refusal',backend,sources['baseline'],'FullyQualifiedName~PackedBoundaryTests.Reduction4096_IsAdmitted',expected=1)
        negative=((BASE if PREDECESSOR is None else PREDECESSOR)/'logs/baseline-refusal.log').read_text(encoding='utf8')
        assert 'Expected: 524288' in negative and 'Actual:   0' in negative,negative
        build('candidate',backend,candidate)
        test('candidate-focused',backend,candidate,'FullyQualifiedName~Packed|FullyQualifiedName~MatMul|FullyQualifiedName~Gemm')
        test('candidate-fallback',backend,candidate,'FullyQualifiedName~PackedBoundaryTests',environment=dict(env,DOTNET_EnableHWIntrinsic='0'))
        build('bridge',bridge/'Bridge.csproj',bridge)
        command('instruction-bridge',['dotnet',bridge/'bin/Release/net10.0/Bridge.dll',sources['baseline']/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0',candidate/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0',BASE/'instructions.json'],ROOT)
        build('cli',Path('src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),candidate)
        tensors=Path('tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')
        build('tensors',tensors,candidate)
        test('candidate-backend',backend,candidate)
        test('candidate-tensors',tensors,candidate)
        state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc()); raise
    finally:
        state.update(complete=True,seconds=time.monotonic()-started);save(BASE/'processes.json',state);own.cpu_affinity(prior_affinity)
    files={}
    for p in sources['baseline'].rglob('*'):
        relative=p.relative_to(sources['baseline'])
        if not p.is_file() or {'bin','obj'}.intersection(relative.parts):continue
        if relative.as_posix() not in ('src/Lokad.Onnx/GraphPacking.cs','tests/Lokad.Onnx.Backend.Tests/FoldBudgetTests.cs'):assert pin(p)==pin(candidate/relative),str(relative)
    for folder in (candidate/'src',candidate/'tests/Lokad.Onnx.Backend.Tests',candidate/'tests/Lokad.Onnx.Tensors.Tests',bridge,TOOLS,BASE/'test-results',BASE/'logs'):
        for p in folder.rglob('*'):
            if p.is_file() and 'obj' not in p.relative_to(folder).parts:files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in (BASE/'candidate.patch',BASE/'source.zip',BASE/'instructions.json',BASE/'processes.json',bridge_source):files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'prepared.json',dict(passed=True,source=commit,files=files,
        failed_predecessor=None if PREDECESSOR is None else pin(PREDECESSOR/'failed-preparation.json'),
        core=pin(candidate/'src/Lokad.Onnx.CLI/bin/Release/net10.0/Lokad.Onnx.dll'),
        scope='Local boundary correctness and source isolation only; no model trajectory, speed claim, AMD result or product promotion'))
    print(json.dumps(dict(passed=True,prepared=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
