"""Freeze a new finite-extreme driver with the exact qualified GraphRaw caller."""
import ast,importlib.util,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-convolution-channel-extremes-amd-v2-20260923'
BUILD=ROOT/'artifacts/pyannote-convolution-channel-build-amd-20260923'
NUM=ROOT/'artifacts/pyannote-convolution-channel-numerics-amd-20260923'
CURRENT=ROOT/'artifacts/parakeet-current-baseline-amd-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('extreme_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
UPSTREAM=[(BUILD,'build','b8573a2d056ffadfad71faaddb53f5787de8d5137c598dc1d5555077c28d1336'),
    (NUM,'numerical','c2be39dcfa6b39a3941ffb776eee8714a96d6d639a7ff6d9a44a5553003081f8'),
    (CURRENT,'current','6c65419f54f93ac43cc9ca26886dcf4bb9b6535e85f40c4db291fb9b9e1ea4bb')]

def previous_closed():
    for folder,label,digest in UPSTREAM:
        assert pin(folder/'closed.json')['sha256']==digest;proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name

def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    copy(ROOT/'global.json',bundle/'source/global.json');copy(TOOLS/'Extremes.cs',bundle/'source/consumer/Extremes.cs')
    (bundle/'source/consumer/Extremes.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AssemblyName>ConvExtremes</AssemblyName><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../../runtime/selected/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../../runtime/selected/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>\n',encoding='utf8')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    for folder,label,digest in UPSTREAM:copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(TOOLS/name,bundle/'tools'/name)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    products=dict(selected={name:pin(CURRENT/'collected/runtimes/current'/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},candidate=read(BUILD/'analysis.json')['built'])
    save(bundle/'stage.json',dict(passed=True,products=products,probe=read(NUM/'analysis.json')['consumers']['raw'],files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))

if __name__=='__main__':prepare()
