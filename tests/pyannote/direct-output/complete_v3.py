"""Correct scalar bias NaN ordering; preserve both earlier failures and all gates."""
import sys
import common

common.BASE=common.ROOT / 'artifacts/pyannote-direct-output-v3-20260922'
common.REMOTE='/dev/shm/lokad-pyannote-direct-output-v3-20260922'
common.monitor.BASE=common.BASE


def scalar_sources():
    from generate_v3 import generate
    source=(common.ROOT / 'src/Lokad.Onnx/MathOps.cs').read_text(encoding='utf-8-sig')
    def extract(name):
        signature='public unsafe static void '+name+'('
        assert source.count(signature)==1
        start=source.index(signature); brace=source.index('{',start); end=brace+1; depth=1
        while depth:
            depth+=(source[end]=='{')-(source[end]=='}'); end+=1
        return source[start:end]
    normal,_=generate(source)
    assert normal.count('if (Avx2.IsSupported)')==1 and normal.count('static class DirectOutput\n{')==1
    scalar=normal.replace('if (Avx2.IsSupported)','if (UseMaskedTail)').replace('static class DirectOutput\n{',
        'static class ScalarDirectOutput\n{\n    static bool UseMaskedTail => false;')
    originals=[extract('mm_unsafe_vectorized_intrinsics_3x4packed'),extract('mm_unsafe_vectorized_intrinsics_2x4packed_bump')]
    assert all(s.count('if (Avx2.IsSupported)')==1 for s in originals)
    header=normal[:normal.index('static class DirectOutput')]
    baseline=header+'static class ScalarOriginal\n{\n    static bool UseMaskedTail => false;\n'+ '\n'.join(s.replace('if (Avx2.IsSupported)','if (UseMaskedTail)') for s in originals)+'\n}\n'
    consumer=(common.TOOLS / 'Probe.cs').read_text(encoding='utf8')
    for old,new,count in [('MathOps.mm_unsafe_vectorized_intrinsics_3x4packed','ScalarOriginal.mm_unsafe_vectorized_intrinsics_3x4packed',1),
        ('MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump','ScalarOriginal.mm_unsafe_vectorized_intrinsics_2x4packed_bump',2),
        ('DirectOutput.Multiply','ScalarDirectOutput.Multiply',1)]:
        assert consumer.count(old)==count; consumer=consumer.replace(old,new)
    return dict(scalar=scalar,baseline=baseline,consumer=consumer)


def prepare():
    import shutil,tarfile,traceback
    from prepare import geometry
    from generate_v3 import generate
    c=common; original=c.ROOT / 'artifacts/pyannote-direct-output-20260922'
    old=c.read(original / 'preparation.json')
    assert old['complete'] and old['code']==1 and old['runs'][-1]['name']=='validate-local-no-avx2'
    c.terminal(old['supervisor'])
    for run in old['runs']:
        assert run['complete']
        for pid,birth in run['members'].items(): c.terminal(dict(pid=int(pid),birth=birth))
    assert [r['code'] for r in old['runs'][:3]]==[0,0,0]
    assert 'FMA required' in (original / 'logs/validate-local-no-avx2.log').read_text()
    assert not c.BASE.exists()
    census=c.ROOT / 'artifacts/pyannote-convolution-epilogue-20260921'
    assert c.pin(census / 'closed.json')['sha256']=='aaf2a6457d49f087b536b1966427ce1ab580554a568652f79c23a40b85ea4b84'
    c.verify(c.read(census / 'closed.json')['files'])
    shape_data=geometry(c.read(census / 'analysis.json'))
    expected=dict(**shape_data,census=c.pin(census / 'closed.json'))
    assert c.read(original / 'payload/shapes.json')==expected
    normal,method=generate((c.ROOT / 'src/Lokad.Onnx/MathOps.cs').read_text(encoding='utf-8-sig'))
    from generate import generate as original_generate
    assert original_generate((c.ROOT / 'src/Lokad.Onnx/MathOps.cs').read_text(encoding='utf-8-sig'))[0]==(original / 'consumer/DirectOutput.cs').read_text(encoding='utf8')
    assert (c.TOOLS / 'Probe.cs').read_bytes()==(original / 'consumer/Probe.cs').read_bytes()
    prior_result=c.read(original / 'output/validate-local.json')
    assert prior_result['passed'] and len(prior_result['records'])==2882
    assert c.pin(original / 'payload/runtime/DirectOutputProbe.dll')['sha256']==prior_result['executable']
    assert c.pin(original / 'payload/runtime/Lokad.Onnx.dll')['sha256']==c.CORE
    c.BASE.mkdir()
    for name in ['logs','consumer','output','payload/runtime','payload/tools']: (c.BASE / name).mkdir(parents=True,exist_ok=True)
    payload=c.BASE / 'payload'
    for p in (original / 'payload/runtime').iterdir(): shutil.copy2(p,payload / 'runtime' / p.name)
    shutil.copy2(original / 'payload/shapes.json',payload / 'shapes.json')
    (c.BASE / 'consumer/DirectOutput.cs').write_text(normal,encoding='utf8')
    shutil.copy2(original / 'consumer/original-method.txt',c.BASE / 'consumer/original-method.txt')
    sources=scalar_sources()
    for name,key in [('ScalarDirectOutput.cs','scalar'),('ScalarOriginal.cs','baseline'),('Probe.cs','consumer')]:
        (c.BASE / 'consumer' / name).write_text(sources[key],encoding='utf8')
    project=c.BASE / 'consumer/Probe.csproj'
    project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AllowUnsafeBlocks>true</AllowUnsafeBlocks><EnableDefaultCompileItems>false</EnableDefaultCompileItems><AssemblyName>ScalarTailProbe</AssemblyName><Nullable>enable</Nullable><NuGetAudit>false</NuGetAudit></PropertyGroup><ItemGroup><Compile Include="Probe.cs"/><Compile Include="ScalarDirectOutput.cs"/><Compile Include="ScalarOriginal.cs"/><Reference Include="Lokad.Onnx"><HintPath>{payload / 'runtime/Lokad.Onnx.dll'}</HintPath></Reference></ItemGroup></Project>''',encoding='utf8')
    normal_folder=c.BASE / 'normal'; normal_folder.mkdir()
    (normal_folder / 'Probe.cs').write_text((c.TOOLS / 'Probe.cs').read_text(encoding='utf8'),encoding='utf8')
    (normal_folder / 'DirectOutput.cs').write_text(normal,encoding='utf8')
    normal_project=normal_folder / 'Probe.csproj'
    normal_project.write_text(project.read_text(encoding='utf8').replace('ScalarTailProbe','DirectOutputProbe').replace('<Compile Include="ScalarDirectOutput.cs"/><Compile Include="ScalarOriginal.cs"/>','<Compile Include="DirectOutput.cs"/>'),encoding='utf8')
    state=c.new_state(); path=c.BASE / 'preparation.json'; c.save(path,state)
    try:
        flags=c.monitor.FLAGS+['-p:NuGetAudit=false']
        for role,proj,assembly_name in [('normal',normal_project,'DirectOutputProbe'),('scalar',project,'ScalarTailProbe')]:
            for name,command in [(role+'-restore',['dotnet','restore',proj,*flags,'--source',c.FEED,'--packages',c.BASE / 'packages']),
                (role+'-build',['dotnet','build',proj,'-c','Release',*flags,'--no-restore','--disable-build-servers'])]:
                c.monitor.worker(state,path,name,command,c.ROOT,[0],8,4,900,True,None)
            binary=proj.parent / 'bin/Release/net10.0'
            assert c.pin(binary / 'Lokad.Onnx.dll')['sha256']==c.CORE
            for suffix in ['dll','deps.json','runtimeconfig.json']:
                shutil.copy2(binary / (assembly_name+'.'+suffix),payload / 'runtime' / (assembly_name+'.'+suffix))
        for name,assembly in [('validate-local','DirectOutputProbe.dll'),('validate-local-scalar-tail','ScalarTailProbe.dll')]:
            c.monitor.worker(state,path,name,['dotnet',payload / 'runtime' / assembly,payload / 'shapes.json','validate',
                c.BASE / 'output' / (name+'.json')],c.ROOT,[0],8,4,900,False,c.BASE / 'output')
            result=c.read(c.BASE / 'output' / (name+'.json')); assert result['passed'] and len(result['records'])==2882
        remote=(c.TOOLS / 'remote.py').read_text(encoding='utf8')
        changes=[("str(BASE / 'runtime/DirectOutputProbe.dll')", "str(BASE / ('runtime/ScalarTailProbe.dll' if name=='validate-scalar-tail' else 'runtime/DirectOutputProbe.dll'))"),
            ("name if name.startswith('validate') else name.split('-')[0]", "'validate' if name.startswith('validate') else name.split('-')[0]"),
            ("                if name=='validate-no-avx2': env['DOTNET_EnableAVX2']='0'\n", ''),
            ("        if name=='validate-no-avx2': assert result['flags']==['DOTNET_EnableAVX2'] and not result['avx2'] and not result['avx512']\n        else: assert result['flags']==[] and result['avx2'] and result['avx512']", "        assert result['flags']==[] and result['avx2'] and result['avx512']")]
        for before,after in changes:
            assert remote.count(before)==1,before; remote=remote.replace(before,after)
        (payload / 'tools/remote.py').write_text(remote,encoding='utf8')
        external=c.read(c.ROOT / 'artifacts/pyannote-amd-profile-runtime-20260922.json')
        spec=dict(files={p.relative_to(payload).as_posix():c.pin(p) for p in payload.rglob('*') if p.is_file()},external=external['files'],
            boot_time=external['boot_time'],core=c.CORE,consumer=c.pin(payload / 'runtime/DirectOutputProbe.dll'),
            scalar_consumer=c.pin(payload / 'runtime/ScalarTailProbe.dll'),jobs=['validate','validate-scalar-tail','baseline-a','candidate-a','candidate-b','baseline-b'],
            gates=dict(process_max_min=1.10,geomean_candidate_baseline=.95,max_shape_candidate_baseline=1.05),
            limits=dict(seconds=900,preflight_available=8*1024**3,preflight_tmpfs=3*1024**3,rss=2*1024**3,
                available=1024**3,tmpfs=1024**3,artifacts=1024**3))
        c.save(payload / 'payload.json',spec)
        with tarfile.open(c.BASE / 'payload.tar.gz','w:gz') as tar:
            for p in sorted(payload.rglob('*')):
                if p.is_file(): tar.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
        pins={c.rel(p):c.pin(p) for folder in [c.TOOLS,c.BASE / 'consumer',normal_folder,payload,original,c.ROOT / 'artifacts/pyannote-direct-output-v2-20260922'] for p in folder.rglob('*')
            if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(folder).parts)}
        for p in [c.ROOT / 'src/Lokad.Onnx/MathOps.cs',c.ROOT / 'src/Lokad.Onnx/Zzz.ConvPortableRows.cs',c.ROOT / 'src/Lokad.Onnx/TensorOps.ConvPool.cs',
            census / 'closed.json',census / 'analysis.json',c.MONITOR,c.ROOT / 'artifacts/pyannote-amd-profile-runtime-20260922.json',
            c.BASE / 'output/validate-local.json',c.BASE / 'output/validate-local-scalar-tail.json']:
            pins[c.rel(p)]=c.pin(p)
        c.save(c.BASE / 'prepared.json',dict(passed=True,files=pins,archive=c.pin(c.BASE / 'payload.tar.gz'),payload=c.pin(payload / 'payload.json')))
        state['code']=0
        print(dict(passed=True,cases_per_mode=2882,payload=c.pin(payload / 'payload.json')),flush=True)
    except BaseException:
        state.update(code=1,error=traceback.format_exc()); raise
    finally:
        state['complete']=True; c.save(path,state)


def audit():
    path=common.TOOLS / 'audit.py'; source=path.read_text(encoding='utf8')
    source=source.replace('from generate import generate','from generate_v3 import generate')
    source=source.replace("['restore','build','validate-local','validate-local-no-avx2']","['normal-restore','normal-build','scalar-restore','scalar-build','validate-local','validate-local-no-avx2']")
    source=source.replace('validate-local-no-avx2','validate-local-scalar-tail').replace('validate-no-avx2','validate-scalar-tail')
    source=source.replace("no_avx2=name.endswith('no-avx2')", "no_avx2=False; scalar=name.endswith('scalar-tail')")
    source=source.replace("r['executable']==spec['consumer']['sha256']", "r['executable']==spec['scalar_consumer' if scalar else 'consumer']['sha256']")
    source=source.replace("('validate-scalar-tail' if no_avx2 else name.split('-')[0])", "name.split('-')[0]")
    namespace=dict(__name__='scalar_tail_audit',__file__=str(path))
    exec(compile(source,str(path),'exec'),namespace)
    sources=scalar_sources()
    for name,key in [('ScalarDirectOutput.cs','scalar'),('ScalarOriginal.cs','baseline'),('Probe.cs','consumer')]:
        assert (common.BASE / 'consumer' / name).read_text(encoding='utf8')==sources[key]
    namespace['main']()


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect','audit']
    mode=sys.argv[1]
    if mode in ['prepare','audit']: globals()[mode]()
    else:
        import transport
        getattr(transport,mode)()
