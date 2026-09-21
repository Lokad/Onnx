"""Build isolated convolution dispatch; qualify numerical/fallback contracts and all-method isolation."""
from common import *
import difflib
import xml.etree.ElementTree as ET

def main():
    assert not BASE.exists()
    receipts=[(PRIOR/'closed.json','ca1fa2265c2128c85b28eedb360e4930d23793054d5a2128bcc3fa0588e72458'),
        (KERNEL/'closed.json','04db6e1bfca159925252dbc9e18f93612c5e84f1a7e62a2662555c1f521b53cd')]
    for p,sha in receipts:
        assert pin(p)['sha256']==sha
        closed=read(p);assert closed['passed'];verify(closed['files'])
        for identity in closed['identities']:terminal(identity)
    assert read(KERNEL/'analysis.json')['eligible']
    BASE.mkdir();(BASE/'logs').mkdir()
    source=BASE/'source';shutil.copytree(PRIOR/'source',source,ignore=shutil.ignore_patterns('bin','obj'))
    runtime=BASE/'runtime';shutil.copytree(QUALIFIED/'application-runtime',runtime)
    path=source/'src/Lokad.Onnx/TensorOps.ConvPool.cs';before=path.read_text(encoding='utf8')
    old='                Tensor<float>.MatMul2D(wView, pView, dView, options);'
    assert before.count(old)==2
    new='''                if (!TryConvPortableRows(wView.Buffer.Span, pView.Buffer.Span, dView.Buffer.Span,
                    tileM, tileKg, colCount, options))
                    Tensor<float>.MatMul2D(wView, pView, dView, options);'''
    start=before.index('    static void RunTiledBatchFloat(');end=before.index('    static void RunPointwiseBatchesFloat(',start)
    fragment=before[start:end];assert fragment.count(old)==1
    after=before[:start]+fragment.replace(old,new)+before[end:];path.write_text(after,encoding='utf8')
    helper=source/'src/Lokad.Onnx/Zzz.ConvPortableRows.cs';assert not helper.exists();shutil.copy2(TOOLS/'ConvPortableRows.cs.txt',helper)
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/src/Lokad.Onnx/TensorOps.ConvPool.cs',tofile='b/src/Lokad.Onnx/TensorOps.ConvPool.cs'))
    patch+=''.join(difflib.unified_diff([],helper.read_text().splitlines(True),fromfile='/dev/null',tofile='b/src/Lokad.Onnx/Zzz.ConvPortableRows.cs'))
    (BASE/'candidate.patch').write_text(patch,encoding='utf8')
    shutil.copy2(TOOLS/'ConvPortableRowsTests.cs',source/'tests/Lokad.Onnx.Backend.Tests/ConvPortableRowsTests.cs')
    bridge=BASE/'bridge';shutil.copytree(PRIOR/'bridge',bridge,ignore=shutil.ignore_patterns('bin','obj'))
    p=bridge/'Program.cs';text=p.read_text(encoding='utf8')
    start=text.index('    object? compilerRename = null;');end=text.index('    var removed =',start)
    text=text[:start]+(TOOLS/'InitializerIdentity.cs.txt').read_text()+text[end:]
    start=text.index('    const string tensor =');end=text.index('    if (!allowed)',start)
    text=text[:start]+'''    const string tensor = "Lokad.Onnx.Tensor`1[T]::";
    bool allowed = name == "Lokad.Onnx.Data.dll"
        ? removed.Length == 0 && added.Length == 0 && differences.Length == 0
          && Hash(Path.Combine(before, name)) == Hash(Path.Combine(after, name))
        : removed.Length == 0 && added.Length == 2
          && added.Count(k => k.StartsWith(tensor + "CanUseConvPortableRows::")) == 1
          && added.Count(k => k.StartsWith(tensor + "TryConvPortableRows::")) == 1
          && differences.Length == 1 && differences[0].StartsWith(tensor + "RunTiledBatchFloat::");
'''+text[end:]
    text=text.replace('equal_except_convolution_output_allocation','equal_except_portable_convolution_dispatch');p.write_text(text,encoding='utf8')
    p=source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj';text=p.read_text(encoding='utf8')
    old_runtime=str(PRIOR/'runtime');assert text.count(old_runtime)==6
    p.write_text(text.replace(old_runtime,str(runtime)),encoding='utf8')
    owner=psutil.Process();st=dict(complete=False,code=None,supervisor=dict(pid=owner.pid,birth=owner.create_time()),runs=[])
    state_path=BASE/'preparation.json';save(state_path,st)
    flags=monitor.FLAGS+['-p:NuGetAudit=false']
    def run(name,args,preflight,output):
        monitor.worker(st,state_path,name,args,source,[0],preflight,8,900,True,output);print(name,'passed',flush=True)
    try:
        for name,project in [('core',source/'src/Lokad.Onnx/Lokad.Onnx.csproj'),('bridge',bridge/'Bridge.csproj')]:
            run(name+'-restore',['dotnet','restore',project,*flags,'--source',FEED,'--packages',BASE/'packages'],8,None)
            run(name+'-build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],8,None)
        shutil.copy2(source/'src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll',runtime/'Lokad.Onnx.dll')
        assert pin(runtime/'Lokad.Onnx.Data.dll')==pin(QUALIFIED/'application-runtime/Lokad.Onnx.Data.dll')
        run('instructions',['dotnet',bridge/'bin/Release/net10.0/Bridge.dll',QUALIFIED/'application-runtime',runtime,BASE/'instructions.json'],8,BASE/'bridge/bin')
        assert read(BASE/'instructions.json')['passed']
        project=source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
        run('backend-restore',['dotnet','restore',project,*flags,'--source',FEED,'--packages',BASE/'packages'],8,None)
        run('backend-build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],8,None)
        for name in ['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']:
            assert pin(project.parent/'bin/Release/net10.0'/(name+'.dll'))==pin(runtime/(name+'.dll'))
        files={rel(p):pin(p) for p in [MONITOR,*[p for p,_ in receipts],BASE/'candidate.patch']}
        for folder in [source,runtime,bridge,TOOLS]:
            for p in folder.rglob('*'):
                if p.is_file() and 'obj' not in p.relative_to(folder).parts:files[rel(p)]=pin(p)
        save(BASE/'focused-prepared.json',dict(passed=True,files=files,core=pin(runtime/'Lokad.Onnx.dll'),data=pin(runtime/'Lokad.Onnx.Data.dll')))
        for name,filter,disabled in [
            ('focused','FullyQualifiedName~Conv|FullyQualifiedName~PoolLifetime|FullyQualifiedName~GraphOwnership',False),
            ('hardware-disabled','FullyQualifiedName~ConvPortableRowsTests',True)]:
            original=monitor.clean_env
            if disabled:
                def environment():
                    env=original();env['DOTNET_EnableHWIntrinsic']='0';return env
                monitor.clean_env=environment
            try:run(name,['dotnet','test',project,'-c','Release',*flags,'--no-build','--no-restore','--filter',filter,'--logger','trx;LogFileName='+name+'.trx','--results-directory',BASE/'test-results'],10,BASE/'test-results')
            finally:monitor.clean_env=original
            counters=ET.parse(BASE/'test-results'/(name+'.trx')).find('.//{*}Counters').attrib
            assert int(counters['failed'])==0 and int(counters['passed'])>=50
            save(BASE/(name+'.json'),dict(passed=True,counters=counters))
        verify(files);st['code']=0
        save(BASE/'prepared.json',dict(passed=True,files=files,core=pin(runtime/'Lokad.Onnx.dll'),data=pin(runtime/'Lokad.Onnx.Data.dll'),scope='Focused convolution dispatch qualification only; model/public timing pending.'))
        print(dict(prepared=pin(BASE/'prepared.json'),core=pin(runtime/'Lokad.Onnx.dll')),flush=True)
    except BaseException:
        st.update(code=1,error=traceback.format_exc());raise
    finally:st['complete']=True;save(state_path,st)

if __name__=='__main__':main()
