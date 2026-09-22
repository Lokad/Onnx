"""Freeze a five-replacement replay adapter, exact products and qualified references."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-wide-screen-amd-20260923'
BUILD=ROOT/'artifacts/pyannote-lstm-wide-build-amd-20260922'
SCREEN=ROOT/'artifacts/pyannote-lstm-input-screen-amd-20260922'
CODEGEN=ROOT/'artifacts/pyannote-lstm-wide-codegen-amd-20260923'
FOCUSED=ROOT/'artifacts/pyannote-lstm-wide-focused-amd-20260923'
CURRENT=ROOT/'artifacts/parakeet-current-baseline-amd-20260922'
QUALIFIED=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-v2-20260922'
BRIDGE=ROOT/'artifacts/pyannote-kernel-loop-numerics-amd-v2-20260922'
AMD=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-20260922'
PLATFORM=ROOT/'artifacts/pyannote-lstm-platform-reference-amd-v3-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('wide_replay_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    for folder,digest in [(SCREEN,'ed5edcaf7ac71c89acb57e3d82ecf80c6239d4aa46210a02e0f8b72adc71ccd5'),(BUILD,'cd9d2839d30d55f2a1d020da040163e1920dc97991e30584b4cdc72593e1d338'),
            (FOCUSED,'78463bb45456b6588d81ff5821304a46774d2d7e9cca05c20d2d2a862b4d453f'),
            (CURRENT,'6c65419f54f93ac43cc9ca26886dcf4bb9b6535e85f40c4db291fb9b9e1ea4bb'),
            (QUALIFIED,'bd855eedf6cf5ce8f7f738ce5d8221e47ad502e52f47d44372a4e98cee8eabe7'),
            (BRIDGE,'551fb5db9dd20c9e5e9c3737a4f311779226bdbc92c53f4cff835f6277799dda')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name

    assert pin(CODEGEN/'reconciled-closed.json')['sha256']=='55dfae31762b55fd2c124404506a270662571eaf3aef138bf534e160c0b16f0b'
    for name,wanted in read(CODEGEN/'reconciled-closed.json')['files'].items():assert pin(CODEGEN/name)==wanted,name


def adapted_source():
    source=(ROOT/'tests/pyannote/lstm-input-screen-amd/Screen.cs').read_text()
    before='+(args[3]=="candidate"?8192:0)';assert source.count(before)==1
    return source.replace(before,'+8192')


def prepare():
    assert not BASE.exists();previous_closed();assert (TOOLS/'Screen.cs').read_text()==adapted_source()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    copy(TOOLS/'Screen.cs',bundle/'source/consumer/Screen.cs');copy(ROOT/'global.json',bundle/'source/global.json')
    project='<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AssemblyName>LstmScreen</AssemblyName><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../../runtime/selected/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../../runtime/selected/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>'
    (bundle/'source/consumer/Screen.csproj').write_text(project,encoding='utf8')
    for p in (BRIDGE/'bundle/bridge').iterdir():
        if p.is_file():copy(p,bundle/'bridge'/p.name)
    for folder,label in [(SCREEN,'screen'),(BUILD,'build'),(FOCUSED,'focused'),(CURRENT,'current'),(QUALIFIED,'qualified')]:copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    copy(CODEGEN/'reconciled-closed.json',bundle/'evidence/codegen-closed.json')
    copy(ROOT/'tests/pyannote/lstm-input-screen-amd/Screen.cs',bundle/'evidence/original-Screen.cs')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(TOOLS/name,bundle/'tools'/name)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    old=read(QUALIFIED/'payload.json')
    products=dict(selected={n:pin(CURRENT/'collected/runtimes/current'/n) for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},candidate=read(BUILD/'analysis.json')['built'])
    from checks import schedule
    save(bundle/'schedule.json',schedule(read(QUALIFIED/'collected/references/capture.json')))
    stage=dict(passed=True,products=products,previous_consumer=read(SCREEN/'collected/built.json')['consumer'],files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR,ROOT/'tests/pyannote/lstm-input-blocks-amd-v2/audit.py']:
        if p.is_file():originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),products=products)))


if __name__=='__main__':prepare()
