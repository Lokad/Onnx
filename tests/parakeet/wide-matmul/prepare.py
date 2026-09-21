"""Bind four qualified encoder requests and build only the probe assembly."""
import ast
import os
import shutil
import subprocess
from common import *


def main():
    assert pin(TRACE/'closed.json')['sha256']=='6d8ce878f99acf291cb20348bba948bd1de7f128298c774cf94adf84e855700d'
    closed=read(TRACE/'closed.json'); assert closed['passed']
    source_spec=read(ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/parakeet.json')
    result=read(TRACE/'trace-output/result.json'); assert result['passed']
    selected=[source_spec['cases'][i] for i in [6,0,3,13]]
    assert [c['expected']['encoded_frames'] for c in selected]==[51,106,167,225]
    rows=[read(TRACE/'trace-output'/name) for name in result['call_files'][:1240]]
    files={}; cases=[]
    def original(p):
        name=p.relative_to(ROOT).as_posix(); wanted=closed['files'][name]; assert pin(p)==wanted;files[name]=wanted
    def tensor(v):
        p=TRACE/'trace-output'/v['file']; original(p)
        return dict(path=p.relative_to(ROOT).as_posix(),**pin(p),shape=v['shape'],dtype=v['dtype'])
    for c in selected:
        matches=[r for r in rows if r['name']==c['name'] and r['graph']=='encoder'];assert len(matches)==1;r=matches[0]
        original(TRACE/'trace-output'/result['call_files'][rows.index(r)])
        cases.append(dict(name=c['name'],frames=c['expected']['encoded_frames'],
            inputs={n:tensor(v) for n,v in r['inputs'].items()},outputs={n:tensor(v) for n,v in r['outputs'].items()}))
    for v in source_spec['models'].values(): original(ROOT/v['path'])
    BASE.mkdir(); (BASE/'source').mkdir()
    save(BASE/'capture-manifest.json',dict(models=list(source_spec['models'].values()),encoder=source_spec['models']['encoder-model.onnx'],cases=cases))
    for p in TOOLS.iterdir():
        if p.suffix in ('.cs','.csproj'): shutil.copy2(p,BASE/'source'/p.name)
        if p.suffix=='.py': ast.parse(p.read_text())
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    flags=['--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:UseSharedCompilation=false','-nr:false']
    commands=[['dotnet','restore','Probe.csproj',*flags,'--source',str(ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'),'--packages',str(BASE/'packages'),'-p:NuGetAudit=false'],
        ['dotnet','build','Probe.csproj','-c','Release',*flags,'--no-restore','--disable-build-servers','-p:FrozenProductDirectory='+str(TRACE/'bin')]]
    builds=[]
    for label,command in zip(['restore','build'],commands,strict=True):
        with (BASE/(label+'.log')).open('x') as log: code=subprocess.run(command,cwd=BASE/'source',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
        builds.append(dict(label=label,command=command,code=code));save(BASE/'builds.json',builds);assert code==0,label
    shutil.copytree(BASE/'source/bin/Release/net10.0',BASE/'bin')
    for p in (TRACE/'bin').glob('*.dll'): shutil.copy2(p,BASE/'bin'/p.name)
    for folder in [TOOLS,BASE/'source',BASE/'bin']:
        for p in folder.iterdir():
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in [BASE/'capture-manifest.json',TRACE/'closed.json',TRACE/'trace-output/graphs.json']:files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'prepared.json',dict(passed=True,files=files,builds=builds,scope='Four real encoder operand fixtures; isolated kernel experiment only'))
    print('Prepared',pin(BASE/'prepared.json'),flush=True)


if __name__=='__main__':main()
