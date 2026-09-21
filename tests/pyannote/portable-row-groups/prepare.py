from common import *

def main():
    assert not BASE.exists()
    assert pin(RUNTIME / 'Lokad.Onnx.dll')['sha256'] == CORE
    census = ROOT / 'tests/pyannote/convolution-allocation/observations-20260921.json'
    pins = {rel(p):pin(p) for p in [census, MONITOR, SOURCE/'MathOps.cs', SOURCE/'TensorOps.ConvPool.cs', SOURCE/'TensorOps.MatMul.cs']}
    ort = ROOT / 'artifacts/parakeet-reduction-source-20260921/kernel-sources.json'
    for row in read(ort):
        path = ROOT / row['path']
        assert pin(path) == {k:row[k] for k in ['bytes','sha256']}
        pins[rel(path)] = pin(path)
    pins[rel(ort)] = pin(ort)
    shapes = {}
    for row in read(census)['cases'][0]['rows']:
        m, channels, kh, kw = row['weight_shape']
        if (kh,kw) == (1,1): continue
        n = channels*kh*kw
        columns = math.prod(row['output_shape'][2:])
        block = columns if n*columns <= 65536 else min(columns,max(32,(65536//(n+m))//32*32))
        for k in sorted({block,columns%block} - {0}):
            shapes[(m,n,k)] = dict(m=m,n=n,k=k,iterations=max(16,min(1000000,1000000000//(2*m*n*k))))
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'consumer').mkdir(); (BASE/'output').mkdir()
    save(BASE/'shapes.json',dict(shapes=list(shapes.values()),source=pin(census),scope='All full/final tiled convolution dimensions; excludes pointwise1x1 path.'))
    shutil.copy2(TOOLS/'Probe.cs',BASE/'consumer/Probe.cs')
    runtime = BASE/'runtime'; shutil.copytree(RUNTIME,runtime)
    project = BASE/'consumer/Probe.csproj'
    project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AllowUnsafeBlocks>true</AllowUnsafeBlocks><EnableDefaultCompileItems>false</EnableDefaultCompileItems><AssemblyName>RowGroupProbe</AssemblyName><Nullable>enable</Nullable><NuGetAudit>false</NuGetAudit></PropertyGroup><ItemGroup><Compile Include="Probe.cs"/><Reference Include="Lokad.Onnx"><HintPath>{runtime / 'Lokad.Onnx.dll'}</HintPath></Reference></ItemGroup></Project>''')
    st = state(); path = BASE/'preparation.json'; save(path,st)
    try:
        flags = monitor.FLAGS+['-p:NuGetAudit=false']
        for name,cmd in [
            ('restore',['dotnet','restore',project,*flags,'--source',monitor.FEED,'--packages',BASE/'packages']),
            ('build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'])]:
            monitor.worker(st,path,name,cmd,ROOT,[0],8,4,900,True,None)
        binary=BASE/'consumer/bin/Release/net10.0'
        assert pin(binary/'Lokad.Onnx.dll')['sha256']==CORE
        for suffix in ['dll','deps.json','runtimeconfig.json']: shutil.copy2(binary/('RowGroupProbe.'+suffix),runtime/('RowGroupProbe.'+suffix))
        for folder in [TOOLS,BASE/'consumer',runtime]:
            for p in folder.rglob('*'):
                if p.is_file() and 'obj' not in p.relative_to(folder).parts: pins[rel(p)] = pin(p)
        pins[rel(BASE/'shapes.json')] = pin(BASE/'shapes.json')
        save(BASE/'prepared.json',dict(passed=True,files=pins,core=CORE,executable=pin(runtime/'RowGroupProbe.dll'),
            jobs=['validate','baseline-a','candidate-a','candidate-b','baseline-b'],
            gates=dict(process_max_min=1.10,geomean_candidate_baseline=.95,max_shape_candidate_baseline=1.05),
            scope='Kernel composition eligibility only; not application timing or AMD admission.'))
        st['code']=0
        print(dict(prepared=pin(BASE/'prepared.json'),shapes=len(shapes)),flush=True)
    except BaseException:
        st.update(code=1,error=traceback.format_exc()); raise
    finally:
        st['complete']=True;save(path,st)

if __name__=='__main__': main()
