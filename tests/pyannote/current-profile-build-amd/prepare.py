"""Freeze the current profile consumer with only expected product identities changed."""
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save
from checks import OLD_CORE,OLD_DATA

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-current-profile-build-amd-20260922'
PRIOR=ROOT/'artifacts/pyannote-prepared-profile-amd-v2-20260922'
CURRENT=ROOT/'artifacts/parakeet-current-baseline-amd-20260922'
SCREEN=ROOT/'artifacts/pyannote-kernel-loop-screen-amd-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('current_profile_build_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    for folder,digest,root_relative in [
        (PRIOR,'87446536bf2ca97f398ec2407c57269e9bc72caf883aaff2a573f520a95a99a4',True),
        (CURRENT,'6c65419f54f93ac43cc9ca26886dcf4bb9b6535e85f40c4db291fb9b9e1ea4bb',False),
        (SCREEN,'926cc187e44395ddeb08beaaaf7d30e9517330ae4b23d70eced938e6a9bb6532',False)]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin((ROOT if root_relative else folder)/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['Program.cs','Diagnostic.cs','NpySupport.cs','SampledAudio.csproj']:copy(PRIOR/'consumer'/name,bundle/'source'/name)
    product={name:pin(CURRENT/'collected/runtimes/current'/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    path=bundle/'source/Program.cs';before=path.read_text(encoding='utf8');after=before
    for old,name in [(OLD_CORE,'Lokad.Onnx.dll'),(OLD_DATA,'Lokad.Onnx.Data.dll')]:
        assert after.count(old)==1;after=after.replace(old,product[name]['sha256'])
    path.write_text(after,encoding='utf8',newline='\n')
    (bundle/'consumer.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='qualified/Program.cs',tofile='current/Program.cs')),encoding='utf8')
    for suffix in ['dll','deps.json','runtimeconfig.json']:copy(PRIOR/'bridge/bin/Release/net10.0'/('Bridge.'+suffix),bundle/'bridge'/('Bridge.'+suffix))
    for p in TOOLS.glob('*.py'):
        ast.parse(p.read_text(),str(p))
        if p.name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(p,bundle/'tools'/p.name)
    for folder,label in [(PRIOR,'prior'),(CURRENT,'current'),(SCREEN,'screen')]:copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    copy(ROOT/'.agent/m24-current-pyannote-attribution-20260922.md',bundle/'prospective-plan.md')
    originals.pop('.agent/m24-current-pyannote-attribution-20260922.md')
    stage=dict(passed=True,product=product,previous_consumer=pin(PRIOR/'payload/runtime/SampledAudio.dll'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),product=product)))


if __name__=='__main__':prepare()
