"""Separate post-capture IL proof and complete affected test suites."""
from common import *
import shutil
import subprocess
import traceback


def main():
    spec=read(BASE/'manifest.json');verify(spec)
    p=read(BASE/'processes.json')
    assert p['complete'] and p['code']==0 and absent(p['supervisor']) and all(absent(r['worker']) for r in p['runs'])
    state_path=BASE/'qualification.json';assert not state_path.exists()
    state=dict(complete=False,code=None,stages=[])
    save(state_path,state)
    flags=['-c','Release','--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
    def call(name,command,cwd):
        with (BASE/'logs'/(name+'.stdout')).open('x') as out,(BASE/'logs'/(name+'.stderr')).open('x') as err:
            code=subprocess.run(command,cwd=cwd,env=clean_env(),stdout=out,stderr=err).returncode
        state['stages'].append(dict(name=name,command=command,cwd=rel(cwd),code=code));save(state_path,state)
        assert code==0,name
        print('qualified',name,flush=True)
    try:
        tool=BASE/'il-project';tool.mkdir()
        for name in ('Compare.csproj','CompareIl.txt'):shutil.copyfile(TOOLS/name,tool/name)
        call('il-build',['dotnet','build','Compare.csproj',*flags],tool)
        binary=tool/'bin/Release/net10.0/Compare.dll'
        qualified=ROOT/'artifacts/whisper-memory-product-v2-20260921/source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
        call('baseline-equivalence',['dotnet',str(binary),str(qualified),str(BASE/'runtimes/baseline'),str(BASE/'baseline-equivalence.json'),'equal'],tool)
        call('candidate-difference',['dotnet',str(binary),str(BASE/'runtimes/baseline'),str(BASE/'runtimes/candidate'),str(BASE/'candidate-difference.json'),'candidate'],tool)
        for name,project in [('backend','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
                             ('tensors','tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')]:
            call(name+'-build',['dotnet','build',project,*flags],BASE/'candidate-source')
            call(name+'-tests',['dotnet','test',project,'--no-build',*flags],BASE/'candidate-source')
        candidate=BASE/'candidate-source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll'
        assert pin(candidate)==spec['cores']['candidate'],'Tests rebuilt different core'
        state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(state_path,state)


if __name__=='__main__':main()
