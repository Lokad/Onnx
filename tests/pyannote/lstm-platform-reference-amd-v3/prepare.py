"""Freeze qualified binaries and exact AMD SDK/native dependencies before workers."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import traceback
from protocol import JOBS, LIMITS, pin, read, save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-platform-reference-amd-v3-20260922'
LOCAL=ROOT/'artifacts/pyannote-lstm-input-blocks-v6-20260922'
PRODUCT=ROOT/'artifacts/pyannote-lstm-input-blocks-v2-20260922'
FIXTURES=ROOT/'artifacts/pyannote-lstm-input-fixtures-v3-20260922'
PREVIOUS=ROOT/'artifacts/pyannote-lstm-platform-reference-amd-v2-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('lstm_amd_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    for folder,name,digest in [(LOCAL,'closed.json','d00b9343dbb5b33eb2bb9db3fa6f139ca12afab910afa370c4421c6bf4a7d683'),
                              (FIXTURES,'closed.json','c64da99fa9e6cbae1af930b6560096c4d9fdd3e7efdf6c07fb6c4e0e53184f7b'),
                              (PREVIOUS,'failure-closed.json','1d30dd35f379245d7288e088447dcdcefc98eea25e59d63c66f4feb99c322224')]:
        assert pin(folder/name)['sha256']==digest
        proof=read(folder/name)
        for key,wanted in proof['files'].items():assert pin(folder/key)==wanted,key
        for identity in proof.get('identities',[]):monitor.terminal(identity)
    receipt=read(PREVIOUS/'collected/collection.json');assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None
    assert receipt['identities'][0]==dict(pid=721597,birth=1790098018.31)
    return receipt['identities'][0]


def prepare():
    assert not BASE.exists();owner=previous_closed();prior=read(PREVIOUS/'payload/payload.json')
    BASE.mkdir();payload=BASE/'payload';payload.mkdir();(payload/'tools').mkdir()
    shutil.copytree(PREVIOUS/'payload/fixtures',payload/'fixtures')
    shutil.copytree(PREVIOUS/'payload/runtime/selected',payload/'runtime/selected')
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        shutil.copy2(PREVIOUS/'collected/built'/('LstmModelReplay.'+suffix),payload/'runtime/selected'/('LstmModelReplay.'+suffix))
    shutil.copytree(PREVIOUS/'collected/selected-256',payload/'retained-256')
    for name in ['protocol.py','checks.py','remote.py','native.py']:shutil.copy2(TOOLS/name,payload/'tools'/name)
    shutil.copy2(ROOT/'PLAN.md',payload/'prospective-plan.md')
    cores=prior['cores'];consumer=read(PREVIOUS/'collected/built.json')['consumer']
    assert pin(payload/'runtime/selected/Lokad.Onnx.dll')==cores['selected'] and pin(payload/'runtime/selected/LstmModelReplay.dll')==consumer
    manifest=dict(passed=True,limits=LIMITS,jobs=JOBS,previous_owner=owner,boot_time=1789634288.0,
        **{k:prior[k] for k in ['external','interpreter','packages']},cores=cores,consumer=consumer,
        files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()},
        scope='Complete remaining selected-platform references; retain already complete AVX2 control and unchanged diagnostic binary. Windows diagnostic maxima permit one serialization ULP; native tensor gates unchanged. No candidate or speed admission.')
    save(payload/'payload.json',manifest)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in [*TOOLS.iterdir(),MONITOR,LOCAL/'closed.json',FIXTURES/'closed.json',PREVIOUS/'failure-closed.json',PREVIOUS/'collected/collection.json'] if p.is_file()}
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(payload.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz'),jobs=JOBS,consumer=consumer)))


if __name__=='__main__':prepare()
