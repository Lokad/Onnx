"""Build the missing isolated CLI, retain the failed suite, finish qualification."""
from common import *
import re
import subprocess
import traceback


def main():
    spec=read(BASE/'manifest.json');verify(spec);old=read(BASE/'qualification.json')
    assert old['complete'] and old['code']==1
    assert [r['name'] for r in old['stages']]==['il-build','baseline-equivalence','candidate-difference','backend-build','backend-tests']
    assert all(r['code']==0 for r in old['stages'][:-1]) and old['stages'][-1]['code']==1
    failed=(BASE/'logs/backend-tests.stderr').read_text();assert failed.count('[FAIL]')==51
    classes=set(re.findall(r'Lokad.Onnx.Backend.Tests.([A-Za-z0-9_]+)\.',failed))
    assert all('Cli' in name for name in classes),classes
    state_path=BASE/'qualification-cli.json';assert not state_path.exists()
    state=dict(complete=False,code=None,predecessor=pin(BASE/'qualification.json'),stages=[])
    save(state_path,state)
    flags=['-c','Release','--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
    def call(name,command):
        with (BASE/'logs'/(name+'.stdout')).open('x') as out,(BASE/'logs'/(name+'.stderr')).open('x') as err:
            code=subprocess.run(command,cwd=BASE/'candidate-source',env=clean_env(),stdout=out,stderr=err).returncode
        state['stages'].append(dict(name=name,command=command,code=code));save(state_path,state)
        assert code==0,name
        print('qualified',name,flush=True)
    try:
        call('cli-build',['dotnet','build','src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj',*flags])
        call('backend-with-cli-tests',['dotnet','test','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj','--no-build',*flags])
        project='tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
        call('tensors-build',['dotnet','build',project,*flags])
        call('tensors-tests',['dotnet','test',project,'--no-build',*flags])
        assert pin(BASE/'candidate-source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll')==spec['cores']['candidate']
        state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(state_path,state)


if __name__=='__main__':main()
