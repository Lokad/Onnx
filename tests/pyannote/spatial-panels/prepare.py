"""Adapt two voice commits onto an isolated current archive; build both cores."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-spatial-panels-20260921'
SOURCE='bb5b092'
PRODUCT=ROOT/'artifacts/whisper-memory-product-v2-20260921/source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
INPUT=ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def git_source(revision,path):
    return subprocess.check_output(['git','show',revision+':'+path],cwd=ROOT).decode('utf8')


def section(text,start,end):
    assert text.count(start)==1,start
    first=text.index(start)
    return text[first:text.index(end,first+len(start))]


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def clean_env():return {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}


def main():
    BASE.mkdir(exist_ok=False);(BASE/'logs').mkdir()
    paths=['src','tests/Lokad.Onnx.Backend.Tests','tests/Lokad.Onnx.Tensors.Tests','tests/Shared',
        'global.json','README.md','CHANGELOG.md','LICENSE.txt','icon.png','docs']
    subprocess.run(['git','archive','--format=zip','-o',str(BASE/'source.zip'),SOURCE,*paths],cwd=ROOT,check=True)
    with zipfile.ZipFile(BASE/'source.zip') as archive:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist())
        for role in ('baseline','candidate'):archive.extractall(BASE/(role+'-source'))
    conv='src/Lokad.Onnx/TensorOps.ConvPool.cs';math='src/Lokad.Onnx/MathOps.cs'
    voice_conv=git_source('e35653a',conv);voice_math=git_source('8af223e',math)
    for name,content in [('voice-ConvPool.cs',voice_conv),('voice-MathOps.cs',voice_math)]:
        (BASE/name).write_text(content,encoding='utf8')
    path=BASE/'candidate-source'/conv;original=path.read_text(encoding='utf-8-sig')
    candidate=original.replace('using System.Runtime.Intrinsics.X86;','using System.Runtime.Intrinsics.X86;\nusing System.Runtime.Intrinsics;')
    anchor='    // Shared Conv2D preparation for PadType padding:'
    assert candidate.count(anchor)==1
    candidate=candidate.replace(anchor,'    // Spatial expansion adapted from voice 8af223e/e35653a; no reduction blocking.\n    const int ConvTileBudgetBytes = 256 * 1024;\n\n'+anchor)
    dispatch=section(voice_conv,'        int tileN = outH * outW;\n        int tileKFull = C * kH * kW;', '        if (dop > 1)')
    dispatch=dispatch.replace(', options, fuseRelu)',', options)')
    anchor='        int patchSize = C * kH * kW * outH * outW;'
    assert candidate.count(anchor)==2
    candidate=candidate.replace(anchor,dispatch+anchor,1)
    methods=section(voice_conv,'    /// <summary>\n    /// Runs convolution in bounded column tiles','    /// <summary>\n    /// Runs 1x1 stride-1 no-pad batches')
    methods=methods.replace(', TensorExecutionOptions options, bool fuseRelu)',', TensorExecutionOptions options)')
    methods=methods.replace(', options, fuseRelu)',', options)')
    methods=methods.replace('os[outRow + j] = fuseRelu && v < 0f ? 0f : v;','os[outRow + j] = v;')
    methods=methods.replace('bias/ReLU epilogue','bias epilogue').replace('keeps the single-pass add order and max, including NaN handling.','keeps the single-pass bias-add order.')
    anchor='    /// <summary>\n    /// Runs 1x1 stride-1 no-pad batches'
    assert candidate.count(anchor)==1
    candidate=candidate.replace(anchor,methods+anchor)
    path.write_text(candidate,encoding='utf8')
    path=BASE/'candidate-source'/math;original_math=path.read_text(encoding='utf-8-sig')
    helper=section(voice_math,'    /// <summary>\n    /// Image to column conversion restricted','    public static unsafe void Im2col(double* src,')
    # Keep this unchecked pointer helper internal, reached only after Conv validation.
    helper=helper.replace('public static unsafe void Im2colRange','internal static unsafe void Im2colRange')
    anchor='    public static unsafe void Im2col(double* src,'
    assert original_math.count(anchor)==1
    changed_math=original_math.replace(anchor,helper+anchor);path.write_text(changed_math,encoding='utf8')
    patch=''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='a/'+conv,tofile='b/'+conv))
    patch+=''.join(difflib.unified_diff(original_math.splitlines(True),changed_math.splitlines(True),fromfile='a/'+math,tofile='b/'+math))
    (BASE/'candidate.patch').write_text(patch,encoding='utf8')
    # Adapt the selected branch tests to master's separate activation contract.
    test=git_source('e35653a','tests/Lokad.Onnx.Backend.Tests/ConvTiledTests.cs')
    test=test.replace('WithBiasAndRelu','WithBias').replace(', true);',');').replace(', false);',');')
    # Restore the oracle helper's Boolean argument while all Conv calls drop it.
    test=test.replace('AssertNear(expected, y.ToArray());','AssertNear(expected, y.ToArray(), false);')
    test=test.replace('TensorExecutionOptions.Auto with { ScratchReporter = acc }','TensorExecutionOptions.Scalar with { ScratchReporter = acc }')
    test=test.replace('1e-4 * (1.0 + System.Math.Abs(e))','1e-4 * System.Math.Max(1.0, System.Math.Abs(e))')
    test='// Adapted from voice 8af223e/e35653a; scratch totals isolated with scalar GEMM.\n'+test
    (BASE/'candidate-source/tests/Lokad.Onnx.Backend.Tests/ConvTiledTests.cs').write_text(test,encoding='utf8')
    shutil.copy2(TOOLS/'SpatialPanelTests.cs',BASE/'candidate-source/tests/Lokad.Onnx.Backend.Tests/SpatialPanelTests.cs')
    flags=['-c','Release','--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
    builds=[]
    def build(name,command,cwd):
        with (BASE/'logs'/(name+'.log')).open('x') as f:
            code=subprocess.run(command,cwd=cwd,env=clean_env(),stdout=f,stderr=subprocess.STDOUT).returncode
        builds.append(dict(name=name,command=command,cwd=str(cwd.relative_to(ROOT)),code=code));save(BASE/'builds.json',builds)
        assert code==0,name
        print('built',name,flush=True)
    for role in ('baseline','candidate'):
        build(role,['dotnet','build','src/Lokad.Onnx/Lokad.Onnx.csproj',*flags],BASE/(role+'-source'))
    build('candidate-tests',['dotnet','build','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',*flags],BASE/'candidate-source')
    build('focused-tests',['dotnet','test','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',*flags,'--no-build',
        '--filter','FullyQualifiedName~Conv|FullyQualifiedName~SpatialPanel'],BASE/'candidate-source')
    consumer=BASE/'consumer';consumer.mkdir()
    source=ROOT/'tests/pyannote/performance-profile/Program.cs'
    text=source.read_text().replace('args.Length != 3','args.Length != 4').replace('root manifest new-output','root manifest new-output expected-core-sha')
    text=text.replace('"d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4"','args[3]')
    text=text.replace('foreach (var c in cases)\n{','for (int applicationPass = 0; applicationPass < 4; applicationPass++)\nforeach (var c in cases)\n{')
    text=text.replace('applications.Add(new { name = c.Name, seconds,','applications.Add(new { name = c.Name, pass = applicationPass, phase = applicationPass == 0 ? "warmup" : "measured", seconds,')
    text=text.replace('applications.Count == 4','applications.Count == 16')
    assert 'applicationPass < 4' in text and 'args[3]' in text
    (consumer/'Program.cs').write_text(text,encoding='utf8')
    project=(ROOT/'tests/pyannote/performance-profile/Profile.csproj').read_text().replace('../../Shared/NpySupport.cs','NpySupport.cs')
    (consumer/'Profile.csproj').write_text(project,encoding='utf8')
    shutil.copy2(ROOT/'tests/Shared/NpySupport.cs',consumer/'NpySupport.cs')
    build('consumer',['dotnet','build','Profile.csproj',*flags,'-p:FrozenProductDirectory='+str(PRODUCT)],consumer)
    cores={}
    for role in ('baseline','candidate'):
        target=BASE/'runtimes'/role;shutil.copytree(consumer/'bin/Release/net10.0',target)
        for path in PRODUCT.glob('*.dll'):shutil.copy2(path,target/path.name)
        source=BASE/(role+'-source')/'src/Lokad.Onnx/bin/Release/net10.0'
        for suffix in ('dll','pdb'):shutil.copy2(source/('Lokad.Onnx.'+suffix),target/('Lokad.Onnx.'+suffix))
        cores[role]=pin(target/'Lokad.Onnx.dll')
    files={}
    for role in ('baseline','candidate'):
        for path in (BASE/(role+'-source')).rglob('*'):
            if path.is_file() and not {'bin','obj'}.intersection(path.relative_to(BASE).parts):files[str(path.relative_to(ROOT))]=pin(path)
    for path in BASE.rglob('*'):
        if path.is_file() and not any(part.endswith('-source') for part in path.relative_to(BASE).parts):files[str(path.relative_to(ROOT))]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    files[str(INPUT.relative_to(ROOT))]=pin(INPUT)
    spec=json.loads(INPUT.read_text())
    for item in list(spec['models'].values())+[c['pcm'] for c in spec['cases']]:
        assert pin(ROOT/item['path'])=={k:item[k] for k in ('bytes','sha256')}
        files[item['path']]=pin(ROOT/item['path'])
    save(BASE/'manifest.json',dict(source=SOURCE,cores=cores,files=files,input=str(INPUT.relative_to(ROOT)),jobs=['baseline','candidate','candidate','baseline']))
    print(json.dumps(dict(manifest=pin(BASE/'manifest.json'),cores=cores,files=len(files))))


if __name__=='__main__':main()
