"""Build a distinct private sharing prototype; never alter the running buffer-reuse artifact."""
from pathlib import Path
import difflib,hashlib,json,shutil,subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/whisper-weight-sharing-20260920'
PRIOR=ROOT/'artifacts/whisper-buffer-reuse-20260920'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)


def run(args,name):
    with (BASE/name).open('x',encoding='utf-8') as f:process=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    assert process.returncode==0,name


def main():
    assert not BASE.exists();BASE.mkdir();source=BASE/'source';source.mkdir()
    inputs={};inherited={}
    for name in ['source.json','cli-source.json']:
        path=PRIOR/name;receipt=json.loads(path.read_text());inputs[path.relative_to(ROOT).as_posix()]=pin(path)
        for relative,wanted in receipt['files'].items():
            old=PRIOR/'source'/relative;assert pin(old)==wanted,relative
            target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,target);inherited[relative]=wanted
    folder=Path(__file__).resolve().parent
    for name,relative in [('WhisperDecoderWeights.cs','src/Lokad.Onnx.Data/WhisperDecoderWeights.cs'),('WhisperDecoderWeightsTests.cs','tests/Lokad.Onnx.Backend.Tests/WhisperDecoderWeightsTests.cs')]:
        path=folder/name;inputs[path.relative_to(ROOT).as_posix()]=pin(path);shutil.copyfile(path,source/relative)
    path=source/'src/Lokad.Onnx.Data/WhisperTranscriber.cs';old=path.read_text(encoding='utf-8')
    assert old.count('    readonly WhisperGeneration generation;')==1
    text=old.replace('    readonly WhisperGeneration generation;', '    readonly WhisperGeneration generation;\n    internal long SharedDecoderWeightBytes { get; }')
    marker='        RequireInputs(pastDecoder, pastNames);';assert text.count(marker)==1
    text=text.replace(marker,marker+'\n        SharedDecoderWeightBytes = WhisperDecoderWeights.Share(firstDecoder, pastDecoder);')
    path.write_text(text,encoding='utf-8')
    (BASE/'WhisperTranscriber.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='buffer-reuse/WhisperTranscriber.cs',tofile='weight-sharing/WhisperTranscriber.cs')),encoding='utf-8')
    census=ROOT/'artifacts/whisper-decoder-weight-census-20260920/census.json'
    inputs[census.relative_to(ROOT).as_posix()]=pin(census);shutil.copyfile(census,BASE/'weight-census.json')
    shutil.copyfile(ROOT/'.agent/m4-whisper-decoder-weight-sharing-20260920.md',BASE/'prospective-plan.md')
    files={p.relative_to(source).as_posix():pin(p) for p in sorted(source.rglob('*')) if p.is_file()}
    write(BASE/'source.json',dict(parent_revision='18e10e3',inherited=inherited,inputs=inputs,files=files))
    flags=['--tl:off','--nologo','-v','minimal','-c','Release']
    run(['dotnet','build',str(source/'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),*flags],'cli-build.log')
    project=source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
    run(['dotnet','build',str(project),*flags],'test-build.log')
    run(['dotnet','test',str(project),*flags,'--no-build','--logger','trx;LogFileName=backend.trx','--results-directory',str(BASE/'test-results')],'tests.log')
    for name,wanted in files.items():assert pin(source/name)==wanted,name
    product=BASE/'product-bin';product.mkdir();built=source/'src/Lokad.Onnx.CLI/bin/Release/net10.0'
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Google.Protobuf.dll','FastBertTokenizer.dll','Lokad.Tokenizers.dll','SixLabors.ImageSharp.dll']:shutil.copyfile(built/name,product/name)
    paths=[BASE/'source.json',BASE/'cli-build.log',BASE/'test-build.log',BASE/'tests.log',BASE/'test-results/backend.trx',*sorted(product.iterdir())]
    value=dict(built=True,tests_passed=True,vm_started=False,files={p.relative_to(BASE).as_posix():pin(p) for p in paths})
    write(BASE/'built.json',value);print(json.dumps(dict(built=True,tests_passed=True,vm_started=False,receipt=pin(BASE/'built.json'))))


if __name__=='__main__':main()
