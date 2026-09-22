"""Copy qualified products and fixtures; stage only a diagnostic consumer build."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,FILTER,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
SOURCE=Path('/dev/shm/lokad-pyannote-lstm-input-blocks-v2-20260922')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=12*1024**3 and psutil.disk_usage(BASE).free>=3*1024**3
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(SOURCE/'payload.json')['sha256']=='9dc81ca197c65ede9d436d996d79268b4b3f496e66dbac72024e9614530d0bde'
    source=read(SOURCE/'payload.json');receipt=read(SOURCE/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert receipt['identities'][0]==dict(pid=722886,birth=1790099220.97)
    for identity in receipt['identities']:assert not live(identity)
    for name,wanted in source['files'].items():assert pin(SOURCE/name)==wanted,name
    for name,wanted in source['external'].items():assert pin(name)==wanted,name
    assert source['cores']==stage['cores'] and source['consumer']==stage['qualified_consumer']
    for name in ['runtime','fixtures','references']:shutil.copytree(SOURCE/name,BASE/name)
    for role in ['selected','candidate']:
        for suffix in ['dll','deps.json','runtimeconfig.json']:(BASE/'runtime'/role/('LstmModelReplay.'+suffix)).unlink()
    consumer=BASE/'consumer';consumer.mkdir();shutil.copy2(BASE/'ModelReplay.cs',consumer/'ModelReplay.cs')
    (consumer/'ModelReplay.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AssemblyName>LstmModelReplay</AssemblyName><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../runtime/selected/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../runtime/selected/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>',encoding='utf8')
    payload=dict(passed=True,jobs=JOBS,filter=FILTER,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        cores=source['cores'],qualified_consumer=source['consumer'],external=source['external'],interpreter=source['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Unchanged qualified products and references, diagnostic flag-only consumer edit, every emitted JIT tier retained; no timing claim.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
