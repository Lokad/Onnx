"""Archive unchanged product, apply one isolated patch, build and freeze."""
from common import *
import difflib
import shutil
import subprocess
import zipfile

ORIGINAL='''                Profiler.StartNodeProfile(node.ID, node.Op, () => string.Join(",", GetInputTensors(node.Inputs).Select(t => t.TensorNameDesc())));'''
CANDIDATE='''                if (profilerScope.Enabled && !profilerScope.WallOnly)
                {
                    // Keep the captured local inside the detailed-only branch:
                    // capturing the foreach variable allocates even when disabled.
                    var profiledNode = node;
                    Profiler.StartNodeProfile(node.ID, node.Op, () => string.Join(",", GetInputTensors(profiledNode.Inputs).Select(t => t.TensorNameDesc())));
                }
                else
                {
                    Profiler.StartNodeProfile(node.ID, node.Op);
                }'''


def main():
    assert not BASE.exists();BASE.mkdir();(BASE/'logs').mkdir()
    state=dict(complete=False,code=None,builds=[]);save(BASE/'builds.json',state)
    try:
        paths=['src','tests/Lokad.Onnx.Backend.Tests','tests/Lokad.Onnx.Tensors.Tests','tests/Shared',
               'global.json','Lokad.Onnx.slnx','README.md','CHANGELOG.md','LICENSE.txt','icon.png','docs']
        subprocess.run(['git','archive','--format=zip','-o',str(BASE/'source.zip'),SOURCE,*paths],cwd=ROOT,check=True)
        with zipfile.ZipFile(BASE/'source.zip') as archive:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist())
            archive.extractall(BASE/'baseline-source');archive.extractall(BASE/'candidate-source')
        baseline=BASE/'baseline-source/src/Lokad.Onnx/ComputationalGraph.cs'
        candidate=BASE/'candidate-source/src/Lokad.Onnx/ComputationalGraph.cs'
        original=baseline.read_text(encoding='utf-8-sig');assert original.count(ORIGINAL)==1
        changed=original.replace(ORIGINAL,CANDIDATE);candidate.write_text(changed,encoding='utf8')
        (BASE/'candidate.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),
            fromfile='a/src/Lokad.Onnx/ComputationalGraph.cs',tofile='b/src/Lokad.Onnx/ComputationalGraph.cs')),encoding='utf8')
        files={}
        def bind(path):files[rel(path)]=pin(path)
        bind(BASE/'source.zip');bind(BASE/'candidate.patch')
        for path in (BASE/'baseline-source').rglob('*'):
            if path.is_file():
                other=BASE/'candidate-source'/path.relative_to(BASE/'baseline-source')
                if path!=baseline:assert pin(path)==pin(other),str(path)
                bind(path);bind(other)
        flags=['-c','Release','--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
        def build(name,command,cwd):
            with (BASE/'logs'/(name+'.stdout')).open('x') as out,(BASE/'logs'/(name+'.stderr')).open('x') as err:
                code=subprocess.run(command,cwd=cwd,env=clean_env(),stdout=out,stderr=err).returncode
            state['builds'].append(dict(name=name,command=command,cwd=rel(cwd),code=code));save(BASE/'builds.json',state)
            assert code==0,name
            print('built',name,flush=True)
        for role in JOBS:
            build(role,['dotnet','build','src/Lokad.Onnx/Lokad.Onnx.csproj',*flags],BASE/(role+'-source'))
        probe=BASE/'probe';probe.mkdir()
        for name in ('Program.cs','Probe.csproj'):shutil.copyfile(TOOLS/name,probe/name);bind(probe/name)
        core_dir=BASE/'baseline-source/src/Lokad.Onnx/bin/Release/net10.0'
        # Class-library build does not always copy its package runtime dependency.
        protobuf=ROOT/'artifacts/parakeet-layer-trace-v2-20260921/bin/Google.Protobuf.dll'
        shutil.copyfile(protobuf,core_dir/'Google.Protobuf.dll');bind(protobuf)
        build('probe',['dotnet','build','Probe.csproj',*flags,'-p:FrozenCoreDirectory='+str(core_dir)],probe)
        cores={};runner=None
        for role in JOBS:
            output=BASE/'runtimes'/role;shutil.copytree(probe/'bin/Release/net10.0',output)
            for suffix in ('dll','pdb'):
                shutil.copyfile(BASE/(role+'-source')/f'src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.{suffix}',output/f'Lokad.Onnx.{suffix}')
            cores[role]=pin(output/'Lokad.Onnx.dll')
            if runner is None:runner=pin(output/'ProfilerAllocation.dll')
            else:assert pin(output/'ProfilerAllocation.dll')==runner
            for path in output.iterdir():
                if path.is_file():bind(path)
        fixtures=ROOT/'artifacts/e5-randomized-processes-20260921/payload/inputs'
        for name in CASES:
            f=fixtures/(name+'.json');s=read(f);assert s['name']==name;bind(f);bind(fixtures/s['reference_file'])
        bind(ROOT/'models/multilingual-e5-small/model.onnx')
        for path in TOOLS.iterdir():
            if path.is_file():bind(path)
        state.update(complete=True,code=0);save(BASE/'builds.json',state)
        bind(BASE/'builds.json')
        for path in (BASE/'logs').iterdir():bind(path)
        write(BASE/'manifest.json',dict(protocol='e5-profiler-allocation-v1',base=rel(BASE),source=SOURCE,
              jobs=JOBS,cases=CASES,warmup=16,observed=16,limits=LIMITS,cores=cores,runner=runner,
              fixtures=rel(fixtures),files=files))
        print(json.dumps(dict(manifest=pin(BASE/'manifest.json'),cores=cores,runner=runner)))
    except BaseException:
        state.update(complete=True,code=1);save(BASE/'builds.json',state);raise


if __name__=='__main__':main()
