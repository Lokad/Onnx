"""Build the full public contract replay with the independently tested snapshot policy."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/whisper-memory-contracts-v2-20260921'
ORIGINAL=ROOT/'artifacts/whisper-memory-contracts-20260920'
PRIOR=ROOT/'artifacts/whisper-recording-v2-20260919'
PRODUCT=ROOT/'artifacts/whisper-weight-sharing-20260920'


def pin(p):
    p=Path(p)
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def main():
    prepared=read(ORIGINAL/'prepared.json');assert prepared['prepared']
    for name,wanted in prepared['files'].items():assert pin(ORIGINAL/name)==wanted,name
    for name,wanted in prepared['inputs'].items():assert pin(ROOT/name)==wanted,name
    transition=ROOT/'artifacts/whisper-weight-sharing-v2-20260920';checked=read(transition/'transition-tests.json');assert checked['passed']
    for name,wanted in checked['files'].items():assert pin(transition/name)==wanted,name
    policy=ROOT/'tests/whisper/weight-sharing-v2/WeightSnapshot.cs'
    assert pin(policy)==pin(transition/'transition-check/WeightSnapshot.cs')
    BASE.mkdir();source=BASE/'source';source.mkdir();product=BASE/'product-bin';product.mkdir()
    for path in (PRODUCT/'product-bin').iterdir():shutil.copyfile(path,product/path.name)
    text=(ORIGINAL/'source/Program.cs').read_text(encoding='utf-8')
    old='var weightsBefore=Weights(model);';assert text.count(old)==1
    text=text.replace(old,old+'\nFile.WriteAllText(Path.Combine(destination,"weights-before.json"),JsonSerializer.Serialize(weightsBefore,new JsonSerializerOptions{WriteIndented=true}));')
    old='Require(JsonSerializer.Serialize(weightsBefore)==JsonSerializer.Serialize(weightsAfter),"Decoder values or shared storage changed");';assert text.count(old)==1
    text=text.replace(old,'File.WriteAllText(Path.Combine(destination,"weights-after.json"),JsonSerializer.Serialize(weightsAfter,new JsonSerializerOptions{WriteIndented=true}));\nWeightSnapshot.Validate(weightsBefore,weightsAfter);')
    text=text.replace('running.ProcessorAffinity.ToInt64()','Affinity(running)')
    marker='static string Sha(string path)';assert text.count(marker)==1
    text=text.replace(marker,'''static long Affinity(Process process)
{
    if(OperatingSystem.IsWindows())return process.ProcessorAffinity.ToInt64();
    if(OperatingSystem.IsLinux())return process.ProcessorAffinity.ToInt64();
    throw new PlatformNotSupportedException();
}
'''+marker)
    (source/'Program.cs').write_text(text,encoding='utf-8')
    shutil.copyfile(policy,source/'WeightSnapshot.cs')
    for name in ['NpySupport.cs','assets.json']:shutil.copyfile(ORIGINAL/'source'/name,source/name)
    shutil.copyfile(ORIGINAL/'source/WhisperMemoryContracts.csproj',source/'WhisperMemoryContractsV2.csproj')
    with (BASE/'build.log').open('x') as log:r=subprocess.run(['dotnet','build',str(source/'WhisperMemoryContractsV2.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release','-o',str(BASE/'bin')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0
    for path in product.iterdir():
        target=BASE/'bin'/path.name
        if target.exists():assert pin(target)==pin(path)
        else:shutil.copyfile(path,target)
    shutil.copyfile(ROOT/'.agent/m4-whisper-memory-contracts-20260920.md',BASE/'prospective-plan.md')
    paths=[*sorted((BASE/'bin').iterdir()),*sorted(source.glob('*.*')),BASE/'build.log',BASE/'prospective-plan.md']
    write(BASE/'prepared.json',dict(prepared=True,original_prepared=pin(ORIGINAL/'prepared.json'),transition_tests=pin(transition/'transition-tests.json'),
        product_closure=pin(PRODUCT/'local-closed.json'),files={p.relative_to(BASE).as_posix():pin(p) for p in paths},
        inputs=prepared['inputs'],completed_requests=13,refusals=16))
    print(json.dumps(dict(prepared=True,receipt=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
