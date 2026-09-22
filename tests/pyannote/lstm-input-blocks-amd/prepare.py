"""Freeze qualified binaries and exact AMD SDK/native dependencies before workers."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from protocol import JOBS, LIMITS, pin, read, save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-20260922'
LOCAL=ROOT/'artifacts/pyannote-lstm-input-blocks-v6-20260922'
PRODUCT=ROOT/'artifacts/pyannote-lstm-input-blocks-v2-20260922'
FIXTURES=ROOT/'artifacts/pyannote-lstm-input-fixtures-v3-20260922'
PREVIOUS=ROOT/'artifacts/pyannote-spatial-weight-screen-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('lstm_amd_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    for folder,digest in [(LOCAL,'d00b9343dbb5b33eb2bb9db3fa6f139ca12afab910afa370c4421c6bf4a7d683'),
                          (FIXTURES,'c64da99fa9e6cbae1af930b6560096c4d9fdd3e7efdf6c07fb6c4e0e53184f7b')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
        for identity in proof['identities']:monitor.terminal(identity)
    monitor.verify(read(LOCAL/'inputs.json')['files']);monitor.verify(read(LOCAL/'binaries.json')['files'])
    assert pin(PREVIOUS/'closed.json')['sha256']=='51a8ae8307eae17ba4f0fed16b5905846ebe8102042f8a581fd3d6f828e282e3'
    receipt=read(PREVIOUS/'collected/collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    owner=receipt['identities'][0];assert owner==dict(pid=719700,birth=1790094036.11)
    return owner


def environment():
    script='''from pathlib import Path
import hashlib,json,sys,os
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil,numpy,onnxruntime
os.sched_setaffinity(0,{0})
assert onnxruntime.__version__=='1.29.0' and psutil.boot_time()==1789634288.0
def pin(p):
 p=Path(p)
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
paths={Path('/home/vermorel/.dotnet/dotnet'),Path(sys.executable)}
for root in [Path('/home/vermorel/.dotnet/sdk/10.0.204'),Path('/home/vermorel/.dotnet/shared/Microsoft.NETCore.App/10.0.8')]:
 assert root.is_dir()
 paths.update(p for p in root.rglob('*') if p.is_file() and p.suffix in ['.dll','.so','.json'])
for module in [psutil,numpy,onnxruntime]:
 paths.update(p for p in Path(module.__file__).parent.rglob('*') if p.is_file() and p.suffix in ['.py','.so','.json'])
for module in list(sys.modules.values()):
 p=Path(getattr(module,'__file__','') or '')
 if p.is_file() and p.suffix in ['.py','.so']:paths.add(p)
for line in Path('/proc/self/maps').read_text().splitlines():
 text=line.split()[-1]
 if text.startswith('/') and '.so' in text and Path(text).is_file():paths.add(Path(text))
print(json.dumps(dict(external={str(p):pin(p) for p in sorted(paths)},interpreter=pin(sys.executable),
 packages={m.__name__:str(Path(m.__file__).resolve()) for m in [psutil,numpy,onnxruntime]})))
'''
    command=['ssh','-i','C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem','-o','BatchMode=yes','-o','ConnectTimeout=20','vermorel@74.178.91.76','python3 -B -']
    result=subprocess.run(command,input=script,text=True,encoding='utf8',capture_output=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0,result.stderr[-3000:]
    return json.loads(result.stdout)


def prepare():
    assert not BASE.exists();owner=previous_closed();env=environment()
    BASE.mkdir();payload=BASE/'payload';payload.mkdir();(payload/'tools').mkdir();(payload/'fixtures').mkdir()
    for role,folder in [('selected','selected-runtime'),('candidate','runtime')]:shutil.copytree(LOCAL/folder,payload/'runtime'/role)
    shutil.copytree(PRODUCT/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0',payload/'backend')
    assert pin(payload/'backend/Lokad.Onnx.dll')==pin(payload/'runtime/candidate/Lokad.Onnx.dll')
    for folder in ['output','native']:shutil.copytree(FIXTURES/folder,payload/'fixtures'/folder)
    for name in ['protocol.py','checks.py','remote.py','native.py']:shutil.copy2(TOOLS/name,payload/'tools'/name)
    shutil.copy2(LOCAL/'test-results/lstm-ordinary.trx',payload/'ordinary.trx');shutil.copy2(ROOT/'PLAN.md',payload/'prospective-plan.md')
    cores={role:pin(payload/'runtime'/role/'Lokad.Onnx.dll') for role in ['selected','candidate']}
    consumer=pin(payload/'runtime/selected/LstmModelReplay.dll');assert consumer==pin(payload/'runtime/candidate/LstmModelReplay.dll')
    manifest=dict(passed=True,limits=LIMITS,jobs=JOBS,previous_owner=owner,boot_time=1789634288.0,**env,cores=cores,consumer=consumer,
        files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()},
        scope='150 LSTM tests at AVX2/AVX512; all captured outputs on both products at both widths plus scalar; fresh AMD ORT1.29 references. No timing or codegen claim.')
    save(payload/'payload.json',manifest)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in [*TOOLS.iterdir(),MONITOR,LOCAL/'closed.json',PRODUCT/'failure-closed.json',FIXTURES/'closed.json',PREVIOUS/'closed.json',PREVIOUS/'collected/collection.json'] if p.is_file()}
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(payload.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz'),cores=cores,jobs=JOBS,external=len(env['external']))))


if __name__=='__main__':prepare()
