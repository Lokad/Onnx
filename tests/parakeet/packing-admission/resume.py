"""Finish source-policy qualification without changing the already checked runtime."""
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from prepare import ROOT,TOOLS,FEED,pin,save,psutil

PRIOR=ROOT/'artifacts/parakeet-packing-admission-v2-20260921'
BASE=ROOT/'artifacts/parakeet-packing-admission-completion-20260921'


def main():
    old=json.loads((PRIOR/'processes.json').read_text());assert old['complete'] and old['code']==1
    identities=[old['supervisor']]+[dict(pid=int(p),birth=b) for r in old['runs'] for p,b in r['members'].items()]
    for identity in identities:
        try:assert psutil.Process(identity['pid']).create_time()!=identity['birth']
        except psutil.NoSuchProcess:pass
    source=PRIOR/'candidate-source';runtime=source/'src/Lokad.Onnx.CLI/bin/Release/net10.0'
    binaries={n:pin(runtime/n) for n in ('Lokad.Onnx.dll','Lokad.Onnx.Data.dll')}
    targets=['tests/Lokad.Onnx.Backend.Tests/PackedBoundaryTests.cs','tests/whisper/weight-sharing/WhisperDecoderWeightsTests.cs']
    BASE.mkdir();(BASE/'logs').mkdir();preserved=BASE/'before';preserved.mkdir()
    for name in targets:
        path=preserved/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,path)
    for name in ('processes.json','instructions.json','candidate.patch'):shutil.copy2(PRIOR/name,preserved/name)
    shutil.copytree(PRIOR/'logs',preserved/'logs');shutil.copytree(PRIOR/'test-results',preserved/'test-results')
    for name in binaries:shutil.copy2(runtime/name,preserved/name)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in preserved.rglob('*') if p.is_file()}
    save(BASE/'failure.json',dict(passed=False,files=files,terminal_identities=identities,
        reason='341 tensors pass; source policy rejects new boundary optional helper plus an existing archived Whisper helper'))
    shutil.copy2(TOOLS/'PackedBoundaryTests.cs',source/targets[0])
    p=source/targets[1];before=p.read_text(encoding='utf8')
    old_text='    static DenseTensor<float> Weight(float[]? values = null, int[]? shape = null) =>'
    new_text='''    static DenseTensor<float> Weight() => Weight(null, null);
    static DenseTensor<float> Weight(float[] values) => Weight(values, null);
    static DenseTensor<float> Weight(int[] shape) => Weight(null, shape);
    static DenseTensor<float> Weight(float[]? values, int[]? shape) =>'''
    assert before.count(old_text)==1;p.write_text(before.replace(old_text,new_text),encoding='utf8')
    patch=''
    for name in targets:
        patch+=''.join(difflib.unified_diff((preserved/name).read_text().splitlines(True),(source/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (BASE/'test-helpers.patch').write_text(patch,encoding='utf8')
    # Use exactly the original bounded monitor, preserving its code in this receipt.
    tool=TOOLS/'prepare.py';original=tool.read_text(encoding='utf8')
    start=original.index('    own = psutil.Process();');end=original.index('    try:\n        backend=Path(',start)
    monitor=textwrap.dedent(original[start:end]);(BASE/'monitor.py').write_text(monitor,encoding='utf8')
    scope=dict(globals(),commit=old['source']);exec(compile(monitor,str(BASE/'monitor.py'),'exec'),scope)
    state=scope['state'];started=scope['started']
    try:
        backend=Path('tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj')
        tensors=Path('tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')
        scope['command']('backend-build',['dotnet','build',backend,'-c','Release',*scope['flags'],'--no-restore','--disable-build-servers'],source)
        for directory in (runtime,source/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0',source/'tests/Lokad.Onnx.Tensors.Tests/bin/Release/net10.0'):
            for name,wanted in binaries.items():assert pin(directory/name)==wanted,str(directory/name)
        scope['test']('candidate-backend',backend,source)
        scope['test']('candidate-tensors',tensors,source)
        for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
        state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state.update(complete=True,seconds=time.monotonic()-started);save(BASE/'processes.json',state);scope['own'].cpu_affinity(scope['prior_affinity'])
    for folder in (BASE,TOOLS,source/'src',source/'tests/Lokad.Onnx.Backend.Tests',source/'tests/Lokad.Onnx.Tensors.Tests'):
        for p in folder.rglob('*'):
            if p.is_file() and 'obj' not in p.relative_to(folder).parts:files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in (PRIOR/'instructions.json',PRIOR/'candidate.patch',PRIOR/'source.zip',PRIOR/'processes.json',source/targets[1]):files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'prepared.json',dict(passed=True,files=files,core=binaries['Lokad.Onnx.dll'],data=binaries['Lokad.Onnx.Data.dll'],
        source=old['source'],runtime=runtime.relative_to(ROOT).as_posix(),failure=pin(BASE/'failure.json'),
        scope='Boundary qualification complete after test-helper-only repairs; exact original candidate Core/Data bytes unchanged'))
    print(json.dumps(dict(passed=True,prepared=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
