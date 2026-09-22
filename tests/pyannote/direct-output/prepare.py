"""Freeze full real output geometry and qualify the new stores before deployment."""
from common import *
from generate import generate
import math
import shutil
import tarfile
import traceback


def geometry(census):
    shapes={}
    for case in census['cases']:
        for row in case['rows']:
            if not row['tiled']: continue
            m=row['weight_shape'][0]//row['attributes']['group']; n=math.prod(row['weight_shape'][1:])
            stride=math.prod(row['output_shape'][2:]); block=row['block_columns']
            for k in row['tile_widths']:
                starts=[start for start in range(0,stride,block) if min(block,stride-start)==k]
                offsets=sorted({min(starts),max(starts)}); key=(m,n,k,stride)
                value=dict(m=m,n=n,k=k,stride=stride,offsets=offsets,timing_start=max(starts),
                    iterations=max(16,min(1000000,1000000000//(2*m*n*k))))
                assert key not in shapes or shapes[key]==value
                shapes[key]=value
    shapes=[shapes[key] for key in sorted(shapes)]
    assert len(shapes)==22 and sum(len(s['offsets']) for s in shapes)==33
    cases=[]
    def add(m,n,k,stride,start,pattern):
        for bias in [False,True]: cases.append(dict(m=m,n=n,k=k,stride=stride,start=start,bias=bias,pattern=pattern))
    for m in [0,2,4,6,8,10,32,64]:
        for n in [0,1,9,63,64,65,129]:
            for k in [0,1,7,8,15,31,32,33,63,64,65]:
                for stride,start in [(k,0),(k+11,3)]: add(m,n,k,stride,start,'finite')
    for s in shapes:
        for start in s['offsets']: add(s['m'],s['n'],s['k'],s['stride'],start,'finite')
    for pattern in ['zero','special']:
        for m in [32,64]:
            for n in [64,65]:
                for k in [0,1,7,8,15,31,32,33,63,64,65]:
                    for stride,start in [(k,0),(k+11,3)]: add(m,n,k,stride,start,pattern)
    assert len(cases)==2882
    return dict(shapes=shapes,cases=cases)


def main():
    assert not BASE.exists()
    census=ROOT / 'artifacts/pyannote-convolution-epilogue-20260921'
    assert pin(census / 'closed.json')['sha256']=='aaf2a6457d49f087b536b1966427ce1ab580554a568652f79c23a40b85ea4b84'
    verify(read(census / 'closed.json')['files'])
    assert pin(PRIOR / 'runtime/Lokad.Onnx.dll')['sha256']==CORE
    shape_data=geometry(read(census / 'analysis.json'))
    source=ROOT / 'src/Lokad.Onnx/MathOps.cs'
    generated,original=generate(source.read_text(encoding='utf-8-sig'))
    BASE.mkdir()
    for name in ['logs','consumer','output','payload/runtime','payload/tools']: (BASE / name).mkdir(parents=True,exist_ok=True)
    payload=BASE / 'payload'; save(payload / 'shapes.json',dict(**shape_data,census=pin(census / 'closed.json')))
    shutil.copy2(TOOLS / 'Probe.cs',BASE / 'consumer/Probe.cs')
    (BASE / 'consumer/DirectOutput.cs').write_text(generated,encoding='utf8')
    (BASE / 'consumer/original-method.txt').write_text(original,encoding='utf8')
    shutil.copy2(PRIOR / 'runtime/Lokad.Onnx.dll',payload / 'runtime/Lokad.Onnx.dll')
    project=BASE / 'consumer/Probe.csproj'
    project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AllowUnsafeBlocks>true</AllowUnsafeBlocks><EnableDefaultCompileItems>false</EnableDefaultCompileItems><AssemblyName>DirectOutputProbe</AssemblyName><Nullable>enable</Nullable><NuGetAudit>false</NuGetAudit></PropertyGroup><ItemGroup><Compile Include="Probe.cs"/><Compile Include="DirectOutput.cs"/><Reference Include="Lokad.Onnx"><HintPath>{payload / 'runtime/Lokad.Onnx.dll'}</HintPath></Reference></ItemGroup></Project>''',encoding='utf8')
    st=new_state(); path=BASE / 'preparation.json'; save(path,st)
    try:
        flags=monitor.FLAGS+['-p:NuGetAudit=false']
        for name,command in [
            ('restore',['dotnet','restore',project,*flags,'--source',FEED,'--packages',BASE / 'packages']),
            ('build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'])]:
            monitor.worker(st,path,name,command,ROOT,[0],8,4,900,True,None)
        binary=BASE / 'consumer/bin/Release/net10.0'
        assert pin(binary / 'Lokad.Onnx.dll')['sha256']==CORE
        for suffix in ['dll','deps.json','runtimeconfig.json']:
            shutil.copy2(binary / ('DirectOutputProbe.'+suffix),payload / 'runtime' / ('DirectOutputProbe.'+suffix))
        for name,mode in [('validate-local','validate'),('validate-local-no-avx2','validate-no-avx2')]:
            original_env=monitor.clean_env
            if mode=='validate-no-avx2':
                def env():
                    value=original_env(); value['DOTNET_EnableAVX2']='0'; return value
                monitor.clean_env=env
            try:
                monitor.worker(st,path,name,['dotnet',payload / 'runtime/DirectOutputProbe.dll',payload / 'shapes.json',
                    mode,BASE / 'output' / (name+'.json')],ROOT,[0],8,4,900,False,BASE / 'output')
            finally: monitor.clean_env=original_env
            result=read(BASE / 'output' / (name+'.json')); assert result['passed'] and len(result['records'])==len(shape_data['cases'])
        shutil.copy2(TOOLS / 'remote.py',payload / 'tools/remote.py')
        external=read(ROOT / 'artifacts/pyannote-amd-profile-runtime-20260922.json')
        remote_spec=dict(files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()},
            external=external['files'],boot_time=external['boot_time'],core=CORE,consumer=pin(payload / 'runtime/DirectOutputProbe.dll'),
            jobs=['validate','validate-no-avx2','baseline-a','candidate-a','candidate-b','baseline-b'],
            gates=dict(process_max_min=1.10,geomean_candidate_baseline=.95,max_shape_candidate_baseline=1.05),
            limits=dict(seconds=900,preflight_available=8*1024**3,preflight_tmpfs=3*1024**3,rss=2*1024**3,
                available=1024**3,tmpfs=1024**3,artifacts=1024**3))
        save(payload / 'payload.json',remote_spec)
        with tarfile.open(BASE / 'payload.tar.gz','w:gz') as tar:
            for p in sorted(payload.rglob('*')):
                if p.is_file(): tar.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
        pins={rel(p):pin(p) for folder in [TOOLS,BASE / 'consumer',payload] for p in folder.rglob('*')
            if p.is_file() and 'obj' not in p.relative_to(folder).parts}
        for p in [source,ROOT / 'src/Lokad.Onnx/Zzz.ConvPortableRows.cs',ROOT / 'src/Lokad.Onnx/TensorOps.ConvPool.cs',
            census / 'closed.json',census / 'analysis.json',MONITOR,ROOT / 'artifacts/pyannote-amd-profile-runtime-20260922.json',
            BASE / 'output/validate-local.json',BASE / 'output/validate-local-no-avx2.json']:
            pins[rel(p)]=pin(p)
        save(BASE / 'prepared.json',dict(passed=True,files=pins,archive=pin(BASE / 'payload.tar.gz'),payload=pin(payload / 'payload.json')))
        st['code']=0
        print(dict(passed=True,cases_per_mode=len(shape_data['cases']),payload=pin(payload / 'payload.json')),flush=True)
    except BaseException:
        st.update(code=1,error=traceback.format_exc()); raise
    finally:
        st['complete']=True; save(path,st)


if __name__=='__main__': main()
