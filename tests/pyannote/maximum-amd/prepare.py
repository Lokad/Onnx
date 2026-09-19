"""Prepare current qualified product DLLs for the closed maximum-diarization oracle."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pin(path):
    return dict(bytes=path.stat().st_size,sha256=sha(path))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    old=root/'artifacts/pyannote-limit-20260919'
    assert sha(old/'long-application-receipt.json')=='e8c8371f887ea0b5a194d9e370dc33c164919c6362af5a0c82358a2f1bce314a'
    receipt=read(old/'long-application-receipt.json')
    for name,digest in receipt['files'].items():assert sha(old/name)==digest,name
    resource=read(old/'default/receipt.json')
    for name,digest in resource['files'].items():assert sha(old/name)==digest,name
    long=read(old/'native-reference/manifest.json')
    assert receipt['fixed_reference_reproduced'] and receipt['public_decisions_passed'] and len(long['files'])==4146
    assert read(old/'native-discovery/manifest.json')['files']==long['files']
    values=0
    for name,item in long['files'].items():
        assert Path(name).name==name and sha(old/'native-reference'/name)==item['sha256'],name
        array=np.load(old/'native-reference'/name,allow_pickle=False,mmap_mode='r')
        assert list(array.shape)==item['shape'] and np.isfinite(array).all(),name
        values+=array.size
    product=root/'artifacts/whisper-recording-v2-20260919'
    assert sha(product/'receipt.json')=='623d1ad90efa120d0910f1f9fad92deaf3d4d9ad8410b74e8c10f16742d66275'
    product_receipt=read(product/'receipt.json')
    assert sha(product/'frozen.json')==product_receipt['files']['frozen.json']
    frozen=read(product/'frozen.json')
    build=read(old/'build.json')
    short=Path(build['reference'])
    assert sha(short/'manifest.json')==build['reference_sha256']
    short_manifest=read(short/'manifest.json');short_case=short_manifest['cases'][0]
    assert short_case['name']=='dialogue-30s' and short_manifest['models']==long['models']
    pcm_path=short/short_case['pcm']
    assert sha(pcm_path)==short_manifest['files'][short_case['pcm']]['sha256']
    pcm=np.load(pcm_path,allow_pickle=False)
    assert pcm.dtype==np.float32 and pcm.shape==(480000,) and np.isfinite(pcm).all()
    repeated=np.tile(pcm,20)
    assert hashlib.sha256(repeated.astype('<f4',copy=False).tobytes()).hexdigest()==receipt['input_sha256']
    assert np.array_equal(repeated.view(np.uint32),np.load(old/'native-reference'/long['cases'][0]['pcm'],allow_pickle=False).view(np.uint32))
    base=args.artifact.resolve();base.mkdir(parents=True,exist_ok=False)
    payload=base/'payload';payload.mkdir()
    files,borrowed={},{}
    remote_whisper='/home/vermorel/Onnx/artifacts/whisper-recording-amd-v2-20260919'
    remote_dialogue='/home/vermorel/Onnx/artifacts/pyannote-dialogue-20260919'
    def add(source,name,borrow=None):
        destination=payload/name;destination.parent.mkdir(parents=True,exist_ok=True)
        assert not destination.exists();shutil.copyfile(source,destination)
        files[name]=pin(source)
        if borrow is not None:borrowed[name]=borrow
    for name,source in [('reference/long-receipt.json',old/'long-application-receipt.json'),
        ('reference/resource-receipt.json',old/'default/receipt.json'),('reference/windows.json',old/'default/result.json'),
        ('reference/long-native.json',old/'native-reference/manifest.json'),('reference/manifest.json',short/'manifest.json'),
        ('reference/product-receipt.json',product/'receipt.json'),('reference/product-frozen.json',product/'frozen.json'),
        ('reference/OriginalProgram.cs',old/'Program.cs')]:add(source,name)
    add(pcm_path,'reference/'+pcm_path.name,remote_dialogue+'/reference/'+pcm_path.name)
    for name,digest in frozen['binaries']['recording'].items():
        if name.endswith(('.exe','.pdb')) or name=='RecordingReplay.dll':continue
        assert sha(product/'recording-bin'/name)==digest
        add(product/'recording-bin'/name,'bin/'+name,remote_whisper+'/bin/'+name)
    program=(old/'Program.cs').read_text(encoding='utf-8')
    original_scope='Finite synthetic maximum-duration resource probe; no native long-request or broad accuracy claim'
    assert program.count(original_scope)==1
    program=program.replace(original_scope,'Finite synthetic maximum-duration AMD API replay; native public reference audited separately; no broad accuracy claim')
    marker='    avx2 = Avx2.IsSupported'
    assert program.count(marker)==1
    program=program.replace(marker,'    flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_") || k.StartsWith("DOTNET_") || k.StartsWith("COMPlus_")).ToDictionary(k => k, Environment.GetEnvironmentVariable),\n'+marker)
    recipe=base/'recipe';recipe.mkdir()
    (recipe/'Program.cs').write_text(program,encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('Evidence.cs'),recipe/'Evidence.cs')
    npy=product/'source/tests/Shared/NpySupport.cs'
    assert sha(npy)==frozen['source']['tests/Shared/NpySupport.cs']
    shutil.copyfile(npy,recipe/'NpySupport.cs')
    core='05884cfd524cc7130321f5dc1bcd0af17dddc7b97e8428d2d2f59e00edb795c2'
    data='27598aa8d8c6b97a1415302cf3aaced1adcf53b20b64734c0c0e047492ca069d'
    write_new(recipe/'reference-pins.json',dict(manifest_sha256=sha(short/'manifest.json'),models=long['models'],
        pcm=dict(name=pcm_path.name,sha256=sha(pcm_path)),core_sha256=core,data_sha256=data))
    (recipe/'LimitProbe.csproj').write_text('''<Project Sdk="Microsoft.NET.Sdk">
<PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings><IsPackable>false</IsPackable></PropertyGroup>
<ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../payload/bin/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Lokad.Onnx.Data"><HintPath>../payload/bin/Lokad.Onnx.Data.dll</HintPath></Reference><EmbeddedResource Include="reference-pins.json" LogicalName="reference-pins.json" /></ItemGroup>
</Project>
''',encoding='utf-8')
    with (base/'build.log').open('x',encoding='utf-8') as log:
        subprocess.run(['dotnet','build',str(recipe/'LimitProbe.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release','-o',str(payload/'bin')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    log=(base/'build.log').read_text(encoding='utf-8');assert '0 Warning(s)' in log and '0 Error(s)' in log,log
    for suffix in ('deps.json','runtimeconfig.json'):
        original=product/'recording-bin'/('RecordingReplay.'+suffix)
        text=original.read_text(encoding='utf-8').replace('RecordingReplay','LimitProbe')
        (payload/'bin'/('LimitProbe.'+suffix)).write_text(text,encoding='utf-8')
    assert sha(payload/'bin/Lokad.Onnx.dll')==core and sha(payload/'bin/Lokad.Onnx.Data.dll')==data
    # Recheck borrowed files after the linker has copied its direct references.
    for name in list(files):assert pin(payload/name)==files[name],name
    for path in (payload/'bin').iterdir():
        if path.suffix in ('.dll','.json') and 'bin/'+path.name not in files:files['bin/'+path.name]=pin(path)
    for path in recipe.iterdir():
        if path.is_file():add(path,'recipe/'+path.name)
    local_models=read(old/'default/complete.json')['command'][3:7]
    remote_models=[
        '/home/vermorel/Onnx/artifacts/segmented-conv-20260918/models/pyannote-segmentation/segmentation/model.onnx',
        '/home/vermorel/Onnx/artifacts/pyannote-embedding-20260919/models/embedding_encoder.onnx',
        '/home/vermorel/Onnx/artifacts/wespeaker-api-20260919/reference/projection.onnx',
        '/home/vermorel/Onnx/artifacts/pyannote-clustering-20260919/reference/prepared.json']
    models={}
    for key,local,remote in zip(('segmentation','encoder','projection','plda'),local_models,remote_models,strict=True):
        assert sha(Path(local))==long['models'][key]
        models[key]=dict(path=remote,**pin(Path(local)))
    support=root/'tests/parakeet/recording-amd/remote.py'
    old_closed=root/'artifacts/parakeet-recording-amd-20260919/closed.json'
    assert sha(old_closed)=='7d2584e5fe5d78cd8b77ab884139c33fce516ac9b8ef85aa470bb4375a2892e0'
    assert pin(support)==read(old_closed)['tracked'][support.relative_to(root).as_posix()]
    add(support,'process_support.py')
    vm_support=root/'tests/whisper/maximum-speech/vm.py'
    assert sha(vm_support)=='78828fa2525529b019fe741e2db831b90c82c911d3f25367e14b84b1756ac45a'
    add(vm_support,'vm_support.py')
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.cs'):add(path,path.name)
    add(root/'.agent/m3-pyannote-maximum-amd-20260919.md','plan.md')
    bundle=dict(schema=1,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),product_source=frozen['commit'],
        original_long_receipt_sha256=sha(old/'long-application-receipt.json'),files=files,borrowed=borrowed,models=models,
        native_arrays_verified=4146,native_values_verified=values,input_sha256=receipt['input_sha256'])
    write_new(payload/'bundle.json',bundle)
    archive=base/'payload.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for path in sorted(payload.rglob('*')):
            if path.is_file() and path.relative_to(payload).as_posix() not in borrowed:
                # Build symbols/apphost are not executable inputs and are omitted.
                name=path.relative_to(payload).as_posix()
                if name=='bundle.json' or name in files:tar.add(path,arcname=name,recursive=False)
    record=dict(**pin(archive),bundle_sha256=sha(payload/'bundle.json'),logical_bytes=sum(p['bytes'] for p in files.values()),borrowed_bytes=sum(files[name]['bytes'] for name in borrowed))
    write_new(base/'preparation.json',record)
    print(record)


if __name__=='__main__':main()
