"""Build a checked, contiguous-copy successor to the qualified spatial candidate."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-spatial-copy-20260921'
OLD=ROOT/'artifacts/pyannote-spatial-panels-20260921'
INPUT=ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
PRODUCT=OLD/'runtimes/candidate'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def clean_env():return {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}


def main():
    receipt=OLD/'qualification-closed.json'
    assert pin(receipt)['sha256']=='91f55c91d3e22d86ecd938a768b9e37d323ce9f762f143494eb02a9d3b2b74c5'
    previous=json.loads(receipt.read_text());assert previous['passed']
    for name,wanted in previous['files'].items():assert pin(ROOT/name)==wanted,name
    BASE.mkdir(exist_ok=False);(BASE/'logs').mkdir()
    source=BASE/'candidate-source'
    shutil.copytree(OLD/'candidate-source',source,ignore=shutil.ignore_patterns('bin','obj'))
    assert (source/'Lokad.Onnx.slnx').is_file()
    changes=[]
    path=source/'src/Lokad.Onnx/MathOps.cs';before=path.read_text(encoding='utf-8-sig')
    first=before.index('    internal static unsafe void Im2colRange(')
    last=before.index('    public static unsafe void Im2col(double* src,',first)
    after=before[:first]+(TOOLS/'Im2colRange.cs.txt').read_text()+before[last:]
    needle='                for (int kk = 0; kk < 4 * Vector256<float>.Count; kk++) d[kk] = src[kk];'
    assert after.count(needle)==1
    after=after.replace(needle,'                new ReadOnlySpan<float>(src, 32).CopyTo(new Span<float>(d, 32));')
    changes.append((path,before,after));path.write_text(after,encoding='utf8')
    path=source/'src/Lokad.Onnx/TensorOps.ConvPool.cs';before=path.read_text(encoding='utf-8-sig');after=before
    needle='        var output = new DenseTensor<float>((ReadOnlySpan<int>)new int[] { N, M, outH, outW });'
    assert after.count(needle)==1
    after=after.replace(needle,needle+'\n        if (output.Length == 0) return output;')
    start=after.index('        int tileN = outH * outW;\n        int tileKFull = C * kH * kW;')
    end=after.index('        if (blockN < tileN)',start)
    after=after[:start]+'        var spatial = PlanConvSpatialScratch(C, kH, kW, M, outH, outW);\n        int tileN = spatial.columns, blockN = spatial.blockColumns;\n'+after[end:]
    needle='        int patchSize = C * kH * kW * outH * outW;'
    after=after.replace(needle,'        int patchSize = spatial.scratchElements;',1)
    # All new tiled arithmetic is checked; avoid overflow in ceiling division.
    after=after.replace('        int blockPatch = C * kH * kW * blockN;','        int blockPatch = checked(C * kH * kW * blockN);')
    after=after.replace('        int blockOut = M * blockN;','        int blockOut = checked(M * blockN);\n        int scratchLength = checked(blockPatch + blockOut);')
    after=after.replace('RentScratch<float>(blockPatch + blockOut, options)','RentScratch<float>(scratchLength, options)')
    after=after.replace('int numBlocks = (tileN + blockN - 1) / blockN;','int numBlocks = 1 + (tileN - 1) / blockN;')
    anchor='    // Shared Conv2D preparation for PadType padding:'
    assert after.count(anchor)==1
    after=after.replace(anchor,(TOOLS/'ConvSpatialPlan.cs.txt').read_text()+anchor)
    changes.append((path,before,after));path.write_text(after,encoding='utf8')
    patch=''.join(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+str(p.relative_to(source)),tofile='b/'+str(p.relative_to(source)))) for p,a,b in changes)
    (BASE/'candidate.patch').write_text(patch,encoding='utf8')
    shutil.copy2(TOOLS/'SpatialCopyTests.cs',source/'tests/Lokad.Onnx.Backend.Tests/SpatialCopyTests.cs')
    # Auto-selected arithmetic lets this concurrency test run without FMA too.
    test=source/'tests/Lokad.Onnx.Backend.Tests/ConvTiledTests.cs';text=test.read_text()
    assert text.count('TensorExecutionOptions.Parallel(2)')==1
    test.write_text(text.replace('TensorExecutionOptions.Parallel(2)','TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 2 }'),encoding='utf8')
    flags=['-c','Release','--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
    builds=[]
    def build(name,command,env):
        with (BASE/'logs'/(name+'.log')).open('x') as log:
            code=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
        builds.append(dict(name=name,command=command,code=code));save(BASE/'builds.json',builds)
        assert code==0,name
        print(name,'passed',flush=True)
    build('backend-build',['dotnet','build','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',*flags],clean_env())
    command=['dotnet','test','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',*flags,'--no-build','--filter','FullyQualifiedName~Conv|FullyQualifiedName~Spatial']
    build('focused-tests',command,clean_env())
    fallback=clean_env();fallback['DOTNET_EnableHWIntrinsic']='0'
    fallback_command=command[:-1]+['FullyQualifiedName~SpatialCopy|FullyQualifiedName~SpatialPanel|FullyQualifiedName~ConvTiled']
    build('focused-fallback-tests',fallback_command,fallback)
    cores={}
    for role in ('baseline','candidate'):
        target=BASE/'runtimes'/role;shutil.copytree(PRODUCT,target)
        if role=='candidate':
            for suffix in ('dll','pdb'):shutil.copy2(source/'src/Lokad.Onnx/bin/Release/net10.0'/('Lokad.Onnx.'+suffix),target/('Lokad.Onnx.'+suffix))
        cores[role]=pin(target/'Lokad.Onnx.dll')
    assert cores['baseline']['sha256']=='52ace94afc0c02f5e9a10cc20d0d68ddb273ee32468f7df88432be6d32aac1b9'
    files={str(receipt.relative_to(ROOT)):pin(receipt)}
    for path in BASE.rglob('*'):
        if path.is_file() and not {'bin','obj'}.intersection(path.relative_to(BASE).parts):files[str(path.relative_to(ROOT))]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    files[str(INPUT.relative_to(ROOT))]=pin(INPUT)
    spec=json.loads(INPUT.read_text())
    for item in list(spec['models'].values())+[c['pcm'] for c in spec['cases']]:
        assert pin(ROOT/item['path'])=={k:item[k] for k in ('bytes','sha256')}
        files[item['path']]=pin(ROOT/item['path'])
    save(BASE/'manifest.json',dict(predecessor=pin(receipt),cores=cores,files=files,input=str(INPUT.relative_to(ROOT)),jobs=['baseline','candidate','candidate','baseline']))
    print(json.dumps(dict(manifest=pin(BASE/'manifest.json'),cores=cores,files=len(files))))


if __name__=='__main__':main()
