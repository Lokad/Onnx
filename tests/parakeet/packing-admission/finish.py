"""Finish suites; the tensor-only project has no Data assembly dependency."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from prepare import ROOT,TOOLS,FEED,pin,save,psutil

SOURCE_ROOT=ROOT/'artifacts/parakeet-packing-admission-v2-20260921'
PRIOR=ROOT/'artifacts/parakeet-packing-admission-completion-20260921'
BASE=ROOT/'artifacts/parakeet-packing-admission-completion-v2-20260921'


def main():
    prior=json.loads((PRIOR/'processes.json').read_text());assert prior['complete'] and prior['code']==1
    assert [(r['name'],r['code']) for r in prior['runs']]==[('backend-build',0)]
    for identity in [prior['supervisor']]+[dict(pid=int(p),birth=b) for r in prior['runs'] for p,b in r['members'].items()]:
        try:assert psutil.Process(identity['pid']).create_time()!=identity['birth']
        except psutil.NoSuchProcess:pass
    source=SOURCE_ROOT/'candidate-source';runtime=source/'src/Lokad.Onnx.CLI/bin/Release/net10.0'
    binaries={n:pin(PRIOR/'before'/n) for n in ('Lokad.Onnx.dll','Lokad.Onnx.Data.dll')}
    for directory in (runtime,source/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'):
        for name,wanted in binaries.items():assert pin(directory/name)==wanted,str(directory/name)
    assert pin(source/'tests/Lokad.Onnx.Tensors.Tests/bin/Release/net10.0/Lokad.Onnx.dll')==binaries['Lokad.Onnx.dll']
    BASE.mkdir();(BASE/'logs').mkdir();monitor=(PRIOR/'monitor.py').read_text(encoding='utf8')
    scope=dict(globals(),commit=prior['source']);exec(compile(monitor,str(PRIOR/'monitor.py'),'exec'),scope)
    state=scope['state'];started=scope['started']
    try:
        scope['test']('candidate-backend',Path('tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),source)
        scope['test']('candidate-tensors',Path('tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'),source)
        state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state.update(complete=True,seconds=time.monotonic()-started);save(BASE/'processes.json',state);scope['own'].cpu_affinity(scope['prior_affinity'])
    files={}
    for folder in (BASE,PRIOR,TOOLS,source/'src',source/'tests/Lokad.Onnx.Backend.Tests',source/'tests/Lokad.Onnx.Tensors.Tests'):
        for p in folder.rglob('*'):
            if p.is_file() and 'obj' not in p.relative_to(folder).parts:files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in (SOURCE_ROOT/'instructions.json',SOURCE_ROOT/'candidate.patch',SOURCE_ROOT/'source.zip',SOURCE_ROOT/'processes.json',source/'tests/whisper/weight-sharing/WhisperDecoderWeightsTests.cs'):
        files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'prepared.json',dict(passed=True,files=files,core=binaries['Lokad.Onnx.dll'],data=binaries['Lokad.Onnx.Data.dll'],
        source=prior['source'],runtime=runtime.relative_to(ROOT).as_posix(),failure=pin(PRIOR/'failure.json'),
        scope='Boundary qualification complete after test-helper-only repairs; original candidate Core/Data bytes unchanged; tensor project depends on Core alone'))
    print(json.dumps(dict(passed=True,prepared=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
