"""Build and qualify the isolated consumer before any VM execution."""
from common import *
from generate import generate
import shutil
import tarfile
import traceback


def main():
    assert not BASE.exists()
    census = ROOT / 'artifacts/pyannote-convolution-epilogue-20260921'
    assert pin(census / 'closed.json')['sha256'] == 'aaf2a6457d49f087b536b1966427ce1ab580554a568652f79c23a40b85ea4b84'
    verify(read(census / 'closed.json')['files'])
    assert pin(PRIOR / 'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    shapes = read(census / 'analysis.json')['all_tile_shapes']
    assert len(shapes) == 22 and len({tuple(s.values()) for s in shapes}) == 22
    for s in shapes: s['iterations'] = max(16, min(1000000, 1000000000 // (2*s['m']*s['n']*s['k'])))
    source = ROOT / 'src/Lokad.Onnx/MathOps.cs'
    generated, original = generate(source.read_text(encoding='utf8-sig'))
    ort = ROOT / 'artifacts/parakeet-matmul-source-20260921/sgemm.cpp'
    assert pin(ort)['sha256'] == '1554aa0b046c3c02734613ef8ab1105bd55449f7d76116d342d17e1c630cb523'
    BASE.mkdir()
    for name in ['logs','consumer','output','payload/runtime','payload/tools']: (BASE / name).mkdir(parents=True, exist_ok=True)
    payload = BASE / 'payload'
    save(payload / 'shapes.json', dict(shapes=shapes, census=pin(census / 'closed.json')))
    shutil.copy2(TOOLS / 'Probe.cs', BASE / 'consumer/Probe.cs')
    (BASE / 'consumer/Block128.cs').write_text(generated, encoding='utf8')
    (BASE / 'consumer/original-method.txt').write_text(original, encoding='utf8')
    shutil.copy2(PRIOR / 'runtime/Lokad.Onnx.dll', payload / 'runtime/Lokad.Onnx.dll')
    project = BASE / 'consumer/Probe.csproj'
    project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AllowUnsafeBlocks>true</AllowUnsafeBlocks><EnableDefaultCompileItems>false</EnableDefaultCompileItems><AssemblyName>ConvolutionReduction</AssemblyName><Nullable>enable</Nullable><NuGetAudit>false</NuGetAudit></PropertyGroup><ItemGroup><Compile Include="Probe.cs"/><Compile Include="Block128.cs"/><Reference Include="Lokad.Onnx"><HintPath>{payload / 'runtime/Lokad.Onnx.dll'}</HintPath></Reference></ItemGroup></Project>''', encoding='utf8')
    st = new_state(); path = BASE / 'preparation.json'; save(path, st)
    try:
        flags = monitor.FLAGS + ['-p:NuGetAudit=false']
        for name, command in [
            ('restore', ['dotnet','restore',project,*flags,'--source',FEED,'--packages',BASE / 'packages']),
            ('build', ['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'])]:
            monitor.worker(st, path, name, command, ROOT, [0], 8, 4, 900, True, None)
        binary = BASE / 'consumer/bin/Release/net10.0'
        assert pin(binary / 'Lokad.Onnx.dll')['sha256'] == CORE
        for suffix in ['dll','deps.json','runtimeconfig.json']:
            shutil.copy2(binary / ('ConvolutionReduction.'+suffix), payload / 'runtime' / ('ConvolutionReduction.'+suffix))
        monitor.worker(st,path,'validate-local',['dotnet',payload / 'runtime/ConvolutionReduction.dll',
            payload / 'shapes.json','validate',BASE / 'output/validate-local.json'],ROOT,[0],8,4,900,False,BASE / 'output')
        result = read(BASE / 'output/validate-local.json'); assert result['passed'] and len(result['records']) == 3014
        for name in ['remote.py']: shutil.copy2(TOOLS / name, payload / 'tools' / name)
        external = read(ROOT / 'artifacts/pyannote-amd-profile-runtime-20260922.json')
        remote_spec = dict(files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()},
            external=external['files'],boot_time=external['boot_time'],core=CORE,
            consumer=pin(payload / 'runtime/ConvolutionReduction.dll'), jobs=['validate','baseline-a','candidate-a','candidate-b','baseline-b'],
            gates=dict(process_max_min=1.10,geomean_candidate_baseline=.95,max_shape_candidate_baseline=1.05),
            limits=dict(seconds=900,preflight_available=8*1024**3,preflight_tmpfs=3*1024**3,rss=2*1024**3,
                available=1024**3,tmpfs=1024**3,artifacts=1024**3))
        save(payload / 'payload.json', remote_spec)
        with tarfile.open(BASE / 'payload.tar.gz','w:gz') as tar:
            for p in sorted(payload.rglob('*')):
                if p.is_file(): tar.add(p, arcname=p.relative_to(payload).as_posix(), recursive=False)
        pins = {rel(p):pin(p) for folder in [TOOLS,BASE / 'consumer',payload] for p in folder.rglob('*')
            if p.is_file() and 'obj' not in p.relative_to(folder).parts}
        for p in [source,ort,census / 'closed.json',census / 'analysis.json',MONITOR,
            ROOT / 'artifacts/pyannote-amd-profile-runtime-20260922.json',BASE / 'output/validate-local.json']:
            pins[rel(p)] = pin(p)
        save(BASE / 'prepared.json', dict(passed=True,files=pins,archive=pin(BASE / 'payload.tar.gz'),payload=pin(payload / 'payload.json')))
        st['code'] = 0
        print(dict(passed=True,validation_cases=len(result['records']),payload=pin(payload / 'payload.json')),flush=True)
    except BaseException:
        st.update(code=1,error=traceback.format_exc()); raise
    finally:
        st['complete'] = True; save(path,st)


if __name__ == '__main__': main()
