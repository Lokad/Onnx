"""Isolate process-global import diagnostics from unrelated parallel tests."""
from common import *
import subprocess
import traceback


def main():
    spec=read(BASE/'manifest.json');verify(spec);old=read(BASE/'qualification-cli.json')
    assert old['complete'] and old['code']==1
    assert [(r['name'],r['code']) for r in old['stages']]==[('cli-build',0),('backend-with-cli-tests',1)]
    failed=(BASE/'logs/backend-with-cli-tests.stderr').read_text()
    assert failed.count('[FAIL]')==1 and 'PackedWeightBudgetTests.DefaultAndInvalidBudgets_HaveExplicitContracts' in failed
    source=BASE/'candidate-source'
    config=source/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0/xunit.runner.json'
    write(config,dict(parallelizeTestCollections=False,maxParallelThreads=1))
    state_path=BASE/'qualification-serial.json';assert not state_path.exists()
    state=dict(complete=False,code=None,predecessor=pin(BASE/'qualification-cli.json'),configuration=pin(config),stages=[])
    save(state_path,state)
    flags=['-c','Release','--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
    def call(name,command):
        with (BASE/'logs'/(name+'.stdout')).open('x') as out,(BASE/'logs'/(name+'.stderr')).open('x') as err:
            code=subprocess.run(command,cwd=source,env=clean_env(),stdout=out,stderr=err).returncode
        state['stages'].append(dict(name=name,command=command,code=code));save(state_path,state)
        assert code==0,name
        print('qualified',name,flush=True)
    try:
        project='tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
        call('import-diagnostic-isolated',['dotnet','test',project,'--no-build','--filter','FullyQualifiedName~PackedWeightBudgetTests.DefaultAndInvalidBudgets_HaveExplicitContracts',*flags])
        call('backend-serial-tests',['dotnet','test',project,'--no-build',*flags])
        project='tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
        call('tensors-build',['dotnet','build',project,*flags])
        call('tensors-tests',['dotnet','test',project,'--no-build',*flags])
        assert pin(source/'src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll')==spec['cores']['candidate']
        state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(state_path,state)


if __name__=='__main__':main()
