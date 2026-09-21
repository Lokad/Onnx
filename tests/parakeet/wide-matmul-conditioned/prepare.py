"""Reuse closed operands and kernels; change only the measurement schedule."""
import ast
import os
import shutil
import subprocess
from common import *

PARENT=ROOT/'artifacts/parakeet-wide-matmul-v3-20260921'


def main():
    assert pin(PARENT/'closed.json')['sha256']=='d68db7bf40bb447aae9079f76812ee508cef1febace481af237164b8e5c46055'
    prior=read(PARENT/'closed.json');assert prior['passed'];verify(prior['files'])
    finish=read(PARENT/'finish-state.json');assert finish['complete'] and finish['passed'] and finish['code']==0
    terminal(finish['supervisor'])
    for identity in prior['identities']:terminal(identity)
    capture=read(PARENT/'capture-closed.json');assert capture['passed'];verify(capture['files'])
    for identity in capture['identities']:terminal(identity)
    for name in ['Blocked.cs','Capture.cs','Program.cs','Probe.csproj']:
        assert (TOOLS/name).read_bytes()==(ROOT/'tests/parakeet/wide-matmul'/name).read_bytes(),name
    BASE.mkdir();(BASE/'source').mkdir();(BASE/'capture').mkdir()
    shutil.copy2(PARENT/'capture/result.json',BASE/'capture/result.json')
    shutil.copy2(PARENT/'probe-manifest.json',BASE/'probe-manifest.json')
    files=dict(capture['files'])
    for p in [BASE/'capture/result.json',BASE/'probe-manifest.json',PARENT/'closed.json',PARENT/'finish-state.json',PARENT/'audit-supervisor.log']:
        files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'capture-closed.json',dict(passed=True,files=files,identities=capture['identities'],reused=True,
        scope='Unchanged parent actual operands; no encoder inference repeated'))
    for p in TOOLS.iterdir():
        if p.suffix in ('.cs','.csproj'):shutil.copy2(p,BASE/'source'/p.name)
        if p.suffix=='.py':ast.parse(p.read_text())
    flags=['--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:UseSharedCompilation=false','-nr:false']
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    commands=[['dotnet','restore','Probe.csproj',*flags,'--source',str(ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'),'--packages',str(BASE/'packages'),'-p:NuGetAudit=false'],
        ['dotnet','build','Probe.csproj','-c','Release',*flags,'--no-restore','--disable-build-servers','-p:FrozenProductDirectory='+str(TRACE/'bin')]]
    builds=[]
    for label,command in zip(['restore','build'],commands,strict=True):
        with (BASE/(label+'.log')).open('x') as log:code=subprocess.run(command,cwd=BASE/'source',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
        builds.append(dict(label=label,command=command,code=code));save(BASE/'builds.json',builds);assert code==0,label
    shutil.copytree(BASE/'source/bin/Release/net10.0',BASE/'bin')
    for p in (TRACE/'bin').glob('*.dll'):shutil.copy2(p,BASE/'bin'/p.name)
    for folder in [TOOLS,BASE/'source',BASE/'bin']:
        for p in folder.iterdir():
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'prepared.json',dict(passed=True,files=files,builds=builds,conditioning_calls=3,measured_rounds=12,
        schedule=[0,1,11,2,10,3,9,4,8,5,7,6],
        identical_control_limits=dict(aggregate_max_min=1.03,each_worker_max_min=1.10),
        product_trial_territory='M>=64 and M%3!=0; original three-row dispatch stays unchanged',
        product_trial_limits=dict(mean_cell_ratio=.95,max_cell_ratio=1.01,max_worker_ratio=1.05),
        scope='Correct timing carryover confound using unchanged real operands and arithmetic'))
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),consumer=pin(BASE/'bin/Probe.dll'))))


if __name__=='__main__':main()
