"""Build the corrected diagnostic without rebuilding or altering the tested product."""
from pathlib import Path
import json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/whisper/memory-contracts'))
from prepare import pin,read,write
BASE=ROOT/'artifacts/whisper-weight-sharing-v2-20260920'
PRODUCT=ROOT/'artifacts/whisper-weight-sharing-20260920'


def main():
    checked=read(BASE/'transition-tests.json');assert checked['passed']
    for name,wanted in checked['files'].items():assert pin(BASE/name)==wanted,name
    closed=read(PRODUCT/'failure-closed.json');assert closed['closure_passed'] and not closed['campaign_passed']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    folder=Path(__file__).parent
    assert pin(folder/'WeightSnapshot.cs')==pin(BASE/'transition-check/WeightSnapshot.cs')
    source=BASE/'consumer';source.mkdir();product=BASE/'product-bin';product.mkdir()
    for p in (PRODUCT/'product-bin').iterdir():shutil.copyfile(p,product/p.name)
    text=(PRODUCT/'consumer/Program.cs').read_text(encoding='utf-8')
    old='var weightsBefore=Weights(whisper);';assert text.count(old)==1
    text=text.replace(old,old+'\nFile.WriteAllText(Path.Combine(output,"weights-before.json"),JsonSerializer.Serialize(weightsBefore,new JsonSerializerOptions{WriteIndented=true}));')
    old='Require(JsonSerializer.Serialize(weightsBefore)==JsonSerializer.Serialize(weightsAfter),"Decoder weights or storage changed");';assert text.count(old)==1
    text=text.replace(old,'File.WriteAllText(Path.Combine(output,"weights-after.json"),JsonSerializer.Serialize(weightsAfter,new JsonSerializerOptions{WriteIndented=true}));\nWeightSnapshot.Validate(weightsBefore,weightsAfter);')
    (source/'Program.cs').write_text(text,encoding='utf-8')
    shutil.copyfile(folder/'WeightSnapshot.cs',source/'WeightSnapshot.cs')
    shutil.copyfile(PRODUCT/'consumer/NpySupport.cs',source/'NpySupport.cs')
    shutil.copyfile(PRODUCT/'consumer/WhisperWeightSharing.csproj',source/'WhisperWeightSharingV2.csproj')
    with (BASE/'consumer-build.log').open('x') as log:r=subprocess.run(['dotnet','build',str(source/'WhisperWeightSharingV2.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release','-o',str(BASE/'bin')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0
    for p in product.iterdir():
        target=BASE/'bin'/p.name
        if target.exists():assert pin(target)==pin(p)
        else:shutil.copyfile(p,target)
    shutil.copyfile(ROOT/'.agent/m4-whisper-sharing-v2-20260920.md',BASE/'prospective-plan.md')
    paths=[*sorted((BASE/'bin').iterdir()),*sorted(source.glob('*.*')),BASE/'consumer-build.log',BASE/'prospective-plan.md',BASE/'transition-tests.json']
    value=dict(prepared=True,vm_started=False,explicit_gc=False,product_source=pin(PRODUCT/'source.json'),product_built=pin(PRODUCT/'built.json'),
        prior_failure=pin(PRODUCT/'failure-closed.json'),transition_tests=pin(BASE/'transition-tests.json'),
        files={p.relative_to(BASE).as_posix():pin(p) for p in paths})
    write(BASE/'prepared.json',value);print(json.dumps(dict(prepared=True,receipt=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
