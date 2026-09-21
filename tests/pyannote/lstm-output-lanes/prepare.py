"""Build recurrent output-lane projections over the qualified convolution successor."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-output-lanes-20260921'
OLD=ROOT/'artifacts/pyannote-spatial-copy-20260921'
INPUT=ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def clean_env():return {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}


def main():
    assert pin(OLD/'manifest.json')['sha256']=='d52719967d47545b65b5e74bc2d37dc3546d2b37d732b6b3008e752e2ca52e1b'
    receipt=OLD/'qualification-closed.json';previous=json.loads(receipt.read_text());assert previous['passed']
    for name,wanted in previous['files'].items():assert pin(ROOT/name)==wanted,name
    analysis=json.loads((OLD/'analysis.json').read_text())
    assert analysis['passed'] and analysis['timing'][0]['ratio']<1,'Do not compound a losing convolution change'
    BASE.mkdir(exist_ok=False);(BASE/'logs').mkdir()
    source=BASE/'candidate-source'
    shutil.copytree(OLD/'candidate-source',source,ignore=shutil.ignore_patterns('bin','obj'))
    assert (source/'Lokad.Onnx.slnx').is_file()
    helper=source/'src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs'
    helper.write_text((TOOLS/'LstmOutputLanes.cs.txt').read_text(),encoding='utf8')
    path=source/'src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs';before=path.read_text(encoding='utf-8-sig')
    needle='        int H = hiddenSize;'
    assert before.count(needle)==1
    after=before.replace(needle,needle+'\n        using var projections = LstmProjectionPanels.Create(ws, rs, inputSize, H, numDirections, seq, opts.Tensor);')
    first=after.index('                    for (int gh = 0; gh < 4 * H; gh++)')
    last=after.index('                    for (int h = 0; h < H; h++)',first)
    old=after[first:last]
    assert old.count('for (int gh')==2 and 'xw[gh] = acc;' in old and 'hr[gh] = acc;' in old
    replacement='''                    if (projections is not null)
                    {
                        projections.Input(d, xs.Slice(xOff, inputSize), xw);
                        projections.Recurrent(d, hv, hr);
                    }
                    else
                    {
'''+''.join('    '+line if line.strip() else line for line in old.splitlines(True))+'''                    }
'''
    after=after[:first]+replacement+after[last:];path.write_text(after,encoding='utf8')
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs',tofile='b/src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs'))
    patch+=''.join(difflib.unified_diff([],helper.read_text().splitlines(True),fromfile='/dev/null',tofile='b/src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs'))
    (BASE/'candidate.patch').write_text(patch,encoding='utf8')
    shutil.copy2(TOOLS/'LstmOutputLaneTests.cs',source/'tests/Lokad.Onnx.Backend.Tests/LstmOutputLaneTests.cs')
    flags=['-c','Release','--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
    builds=[]
    def build(name,command,env):
        with (BASE/'logs'/(name+'.log')).open('x') as log:
            code=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
        builds.append(dict(name=name,command=command,code=code));save(BASE/'builds.json',builds)
        assert code==0,name
        print(name,'passed',flush=True)
    build('backend-build',['dotnet','build','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',*flags],clean_env())
    command=['dotnet','test','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',*flags,'--no-build','--filter','FullyQualifiedName~Lstm']
    build('focused-tests',command,clean_env())
    fallback=clean_env();fallback['DOTNET_EnableHWIntrinsic']='0'
    build('focused-fallback-tests',command[:-1]+['FullyQualifiedName~LstmOutputLane'],fallback)
    cores={}
    for role in ('baseline','candidate'):
        target=BASE/'runtimes'/role;shutil.copytree(OLD/'runtimes/candidate',target)
        if role=='candidate':
            for suffix in ('dll','pdb'):shutil.copy2(source/'src/Lokad.Onnx/bin/Release/net10.0'/('Lokad.Onnx.'+suffix),target/('Lokad.Onnx.'+suffix))
        cores[role]=pin(target/'Lokad.Onnx.dll')
    assert cores['baseline']['sha256']=='a7da535fe19a409af71be752103a2bf1725f66b9e64cda901d6df1ebd20cf832'
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
