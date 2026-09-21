"""Check both independent policies against actual controlled data and damaged snapshots."""
from pathlib import Path
import copy,json,shutil,subprocess,sys
from weight_transition import validate_transition
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/whisper/memory-contracts'))
from prepare import pin,read,write
BASE=ROOT/'artifacts/whisper-weight-sharing-v2-20260920'


def main():
    prior=ROOT/'artifacts/whisper-weight-metadata-20260920';closed=read(prior/'closed.json');assert closed['passed']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    BASE.mkdir();folder=BASE/'transition-check';folder.mkdir();paths=[];accepted=0;rejected=0
    for host in [prior,prior/'amd-v2/collected']:
        for mode in ['unshared','shared']:
            path=host/mode/'worker/result.json';paths.append(path);v=read(path)
            before=copy.deepcopy(v['snapshots'][0]);after=copy.deepcopy(v['snapshots'][-1]);before.pop('stage');after.pop('stage')
            validate_transition(before,after);accepted+=1
            for damage in [lambda v:v['first']['initializers'][0].update(sha256='0'*64),
                lambda v:v['past']['initializers'][0].update(shape=[-1]),lambda v:v.update(unique_arrays=v['unique_arrays']+1),
                lambda v:v.update(unique_payload_bytes=v['unique_payload_bytes']+1),lambda v:v['first'].update(packed_bytes=v['first']['packed_bytes']+1),
                lambda v:v['past']['nodes'][0].update(op='Invalid'),
                lambda v:next(r for r in v['first']['initializers'] if r['name']=='folded:Transpose_1010').update(tensor_name='unexpected')]:
                bad=copy.deepcopy(after);damage(bad)
                try:validate_transition(before,bad)
                except AssertionError:rejected+=1;continue
                raise AssertionError('Corruption accepted')
    for name in ['WeightSnapshot.cs','CheckSnapshot.cs']:shutil.copyfile(Path(__file__).with_name(name),folder/name)
    project='<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings></PropertyGroup></Project>'
    (folder/'CheckSnapshot.csproj').write_text(project,encoding='utf-8')
    with (folder/'build.log').open('x') as log:r=subprocess.run(['dotnet','build',str(folder/'CheckSnapshot.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0
    with (folder/'run.log').open('x') as log:r=subprocess.run(['dotnet',str(folder/'bin/Release/net10.0/CheckSnapshot.dll'),*map(str,paths)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0 and (folder/'run.log').read_text().strip()=='Accepted 4 real transitions; rejected 28 damaged snapshots.'
    result=dict(passed=True,python_accepted=accepted,python_rejected=rejected,csharp_accepted=4,csharp_rejected=28,
        diagnostic_closure=pin(prior/'closed.json'),inputs={p.relative_to(ROOT).as_posix():pin(p) for p in paths},
        files={p.relative_to(BASE).as_posix():pin(p) for p in sorted(folder.rglob('*')) if p.is_file()})
    write(BASE/'transition-tests.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','files']}))


if __name__=='__main__':main()
