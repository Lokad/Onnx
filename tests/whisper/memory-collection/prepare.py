"""Compile a diagnostic consumer; do not launch or alter any VM workload."""
from pathlib import Path
import hashlib,json,shutil,subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/whisper-memory-collection-20260920'


def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def main():
    assert not BASE.exists();BASE.mkdir();source=BASE/'source';source.mkdir();lib=BASE/'lib';lib.mkdir()
    prior=ROOT/'artifacts/audio-amd-comparison-v2-20260920/collected';receipt=json.loads((prior/'collection.json').read_text())
    assert receipt['code']==1 and receipt['terminal']
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Google.Protobuf.dll','FastBertTokenizer.dll','Lokad.Tokenizers.dll','SixLabors.ImageSharp.dll']:
        p=prior/'bin'/name;assert pin(p)==receipt['files']['bin/'+name];shutil.copyfile(p,lib/name)
    program=Path(__file__).with_name('Program.cs');shutil.copyfile(program,source/'Program.cs')
    shutil.copyfile(ROOT/'tests/Shared/NpySupport.cs',source/'NpySupport.cs')
    project='''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><OutputType>Exe</OutputType><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><AllowUnsafeBlocks>true</AllowUnsafeBlocks></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../lib/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Lokad.Onnx.Data"><HintPath>../lib/Lokad.Onnx.Data.dll</HintPath></Reference></ItemGroup></Project>'''
    (source/'WhisperMemoryCollection.csproj').write_text(project,encoding='utf-8')
    shutil.copyfile(ROOT/'.agent/m4-whisper-memory-collection-20260920.md',source/'prospective-plan.md')
    with (BASE/'build.log').open('x',encoding='utf-8') as log:
        result=subprocess.run(['dotnet','build',str(source/'WhisperMemoryCollection.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release','-o',str(BASE/'bin')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,(BASE/'build.log').read_text()
    for p in lib.iterdir():
        target=BASE/'bin'/p.name
        if target.exists():assert pin(target)==pin(p),p.name
        else:shutil.copyfile(p,target)
    value=dict(prepared=True,vm_started=False,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        original_failure=pin(ROOT/'artifacts/audio-amd-comparison-v2-20260920/failure-closed.json'),original_consumer=pin(ROOT/'tests/audio/whisper-comparison/Program.cs'),
        calls=20,collect_after_calls=[8,16,20],configuration='Explicit blocking compacting generation-2 collection; diagnostic only',
        files={p.relative_to(BASE).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()})
    with (BASE/'prepared.json').open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(prepared=True,vm_started=False,files=len(value['files']),receipt=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
