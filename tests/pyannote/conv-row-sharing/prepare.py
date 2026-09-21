"""Prepare the convolution-only AMD row-sharing candidate; qualify local fallback."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-conv-row-sharing-20260921'
OLD=ROOT/'artifacts/pyannote-lstm-output-lanes-20260921'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def clean_env():return {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}


def main():
    receipt=OLD/'qualification-closed.json'
    assert pin(receipt)['sha256']=='cf950ec5cedf702c1af38decc377cd516a5d8b652f77853d03d2c81db0b5bf53'
    previous=json.loads(receipt.read_text());assert previous['passed']
    for name,wanted in previous['files'].items():assert pin(ROOT/name)==wanted,name
    for name,sha,passed in [
        ('pyannote-optimized-meetings-20260921','f9978dd22cec3091e640030e1db8b8e70aeb3c111c43d704c11f679ca5a89c1c','passed'),
        ('pyannote-optimized-parakeet-20260921','57ded6ece06f486c31689ef14143237a0d188bcd76ced094a39932a45c406817','regression_passed')]:
        path=ROOT/'artifacts'/name/'closed.json';assert pin(path)['sha256']==sha
        closed=json.loads(path.read_text());assert closed[passed]
        for file,wanted in closed['files'].items():assert pin(ROOT/file)==wanted,file
    BASE.mkdir(exist_ok=False);(BASE/'logs').mkdir()
    source=BASE/'candidate-source'
    shutil.copytree(OLD/'candidate-source',source,ignore=shutil.ignore_patterns('bin','obj'))
    assert (source/'Lokad.Onnx.slnx').is_file()
    helper=source/'src/Lokad.Onnx/TensorOps.ConvPackedRows.cs'
    helper.write_text((TOOLS/'ConvPackedRows.cs.txt').read_text(),encoding='utf8')
    path=source/'src/Lokad.Onnx/TensorOps.ConvPool.cs';before=path.read_text(encoding='utf-8-sig');after=before
    needle='        int scratchLength = checked(blockPatch + blockOut);'
    assert after.count(needle)==1
    after=after.replace(needle,needle+'''
        int blockPack = ConvPackedScratchLength(M / group, C * kH * kW / group, blockN, scratchLength, options);
        int packOffset = scratchLength;
        scratchLength = checked(scratchLength + blockPack);''')
    old='RunTiledBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, b,'
    assert after.count(old)==2
    after=after.replace(old,'RunTiledBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, new Memory<float>(scratch, packOffset, blockPack), b,')
    old='static void RunTiledBatchFloat(Memory<float> xMem, Memory<float> wMem, Memory<float> bMem, bool hasBias, Memory<float> oMem, float[] scratch, int b,'
    assert after.count(old)==1
    after=after.replace(old,old.replace('float[] scratch, int b,','float[] scratch, Memory<float> packMem, int b,'))
    start=after.index('            for (int g = 0; g < group; g++)',after.index('static void RunTiledBatchFloat('))
    first=after.index('                var wView =',start);last=after.index('                int outBase =',first)
    old=after[first:last];assert old.count('Tensor<float>.MatMul2D')==1
    after=after[:first]+'''                if (!TryConvPackedTile(wMem.Span.Slice(g * tileM * tileKg, tileM * tileKg),
                    patchMem.Span.Slice(g * tileKg * colCount, tileKg * colCount),
                    outMem.Span.Slice(g * tileM * colCount, tileM * colCount), packMem.Span,
                    tileM, tileKg, colCount, options))
                {
'''+''.join('    '+line if line.strip() else line for line in old.splitlines(True))+'''                }
'''+after[last:]
    path.write_text(after,encoding='utf8')
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/src/Lokad.Onnx/TensorOps.ConvPool.cs',tofile='b/src/Lokad.Onnx/TensorOps.ConvPool.cs'))
    patch+=''.join(difflib.unified_diff([],helper.read_text().splitlines(True),fromfile='/dev/null',tofile='b/src/Lokad.Onnx/TensorOps.ConvPackedRows.cs'))
    (BASE/'candidate.patch').write_text(patch,encoding='utf8')
    shutil.copy2(TOOLS/'ConvPackedRowsTests.cs',source/'tests/Lokad.Onnx.Backend.Tests/ConvPackedRowsTests.cs')
    flags=['-c','Release','--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
    builds=[]
    def build(name,command,env):
        with (BASE/'logs'/(name+'.log')).open('x') as log:
            code=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
        builds.append(dict(name=name,command=command,code=code));save(BASE/'builds.json',builds)
        assert code==0,name
        print(name,'passed',flush=True)
    build('backend-build',['dotnet','build','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',*flags],clean_env())
    command=['dotnet','test','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',*flags,'--no-build','--filter',
        'FullyQualifiedName~Conv|FullyQualifiedName~Spatial|FullyQualifiedName~PackedAvx512Row']
    build('focused-tests',command,clean_env())
    fallback=clean_env();fallback['DOTNET_EnableHWIntrinsic']='0'
    build('focused-fallback-tests',command[:-1]+['FullyQualifiedName~ConvPackedRows|FullyQualifiedName~ConvTiled|FullyQualifiedName~Spatial'],fallback)
    runtime=BASE/'runtime';shutil.copytree(OLD/'runtimes/candidate',runtime)
    for suffix in ('dll','pdb'):shutil.copy2(source/'src/Lokad.Onnx/bin/Release/net10.0'/('Lokad.Onnx.'+suffix),runtime/('Lokad.Onnx.'+suffix))
    assert pin(runtime/'Lokad.Onnx.Data.dll')['sha256']=='e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'
    changed=[]
    for path in (source/'src').rglob('*'):
        if not path.is_file() or {'bin','obj'}.intersection(path.relative_to(source).parts):continue
        prior=OLD/'candidate-source'/path.relative_to(source)
        if not prior.exists() or pin(prior)!=pin(path):changed.append(str(path.relative_to(source)).replace('\\','/'))
    assert set(changed)=={'src/Lokad.Onnx/TensorOps.ConvPool.cs','src/Lokad.Onnx/TensorOps.ConvPackedRows.cs'},changed
    files={str(receipt.relative_to(ROOT)):pin(receipt)}
    for path in BASE.rglob('*'):
        if path.is_file() and not {'bin','obj'}.intersection(path.relative_to(BASE).parts):files[str(path.relative_to(ROOT))]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    result=dict(files=files,core=pin(runtime/'Lokad.Onnx.dll'),data=pin(runtime/'Lokad.Onnx.Data.dll'),changed_source=changed,
        scope='Local build, guards and fallback tests only. AVX-512 execution, full affected models, AMD timing and production promotion pending.')
    save(BASE/'prepared.json',result)
    print(json.dumps(dict(files=len(files),prepared=pin(BASE/'prepared.json'),core=result['core'],scope=result['scope'])))


if __name__=='__main__':main()
