"""Freeze a small tool bundle; derive unchanged product fixtures on the authorized VM."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-input-codegen-amd-20260922'
LOCAL=ROOT/'artifacts/pyannote-lstm-input-blocks-v6-20260922'
AMD=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-20260922'
PLATFORM=ROOT/'artifacts/pyannote-lstm-platform-reference-amd-v3-20260922'
QUALIFIED=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-v2-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('lstm_compare_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    for folder,name,digest in [(LOCAL,'closed.json','d00b9343dbb5b33eb2bb9db3fa6f139ca12afab910afa370c4421c6bf4a7d683'),
                              (AMD,'failure-closed.json','32a258cb1da52f0499e022dd3d0aa787db496fd1affe5c43c68a22093ffb9b13'),
                              (PLATFORM,'closed.json','c9d5349ac2305fe776be6fdc0af7623bc287c691c1e06a05185d5fa0af7e9501')]:
        assert pin(folder/name)['sha256']==digest
        proof=read(folder/name)
        for key,wanted in proof['files'].items():assert pin(folder/key)==wanted,key
        for identity in proof.get('identities',[]):monitor.terminal(identity)
    assert pin(QUALIFIED/'closed.json')['sha256']=='bd855eedf6cf5ce8f7f738ce5d8221e47ad502e52f47d44372a4e98cee8eabe7'
    for name,wanted in read(QUALIFIED/'closed.json')['files'].items():assert pin(QUALIFIED/name)==wanted,name
    monitor.verify(read(LOCAL/'inputs.json')['files']);monitor.verify(read(LOCAL/'binaries.json')['files'])


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();(bundle/'tools').mkdir()
    for name in ['protocol.py','checks.py','remote.py','remote_prepare.py']:shutil.copy2(TOOLS/name,bundle/'tools'/name)
    shutil.copy2(ROOT/'PLAN.md',bundle/'prospective-plan.md')
    original=read(QUALIFIED/'payload.json')
    source=(ROOT/'tests/pyannote/lstm-input-blocks-v6/ModelReplay.cs').read_text()
    expected=source.replace('(args[4] == "scalar" && k == "DOTNET_EnableHWIntrinsic"))','(args[4] == "scalar" && k == "DOTNET_EnableHWIntrinsic") || k == "DOTNET_JitDisasm")')
    expected=expected.replace('flags.All(k => Environment.GetEnvironmentVariable(k) == "0")','flags.All(k => Environment.GetEnvironmentVariable(k) == (k == "DOTNET_JitDisasm" ? "LstmProject*" : "0"))')
    assert (TOOLS/'ModelReplay.cs').read_text()==expected
    shutil.copy2(TOOLS/'ModelReplay.cs',bundle/'ModelReplay.cs')
    stage=dict(passed=True,cores=original['cores'],qualified_consumer=original['consumer'],files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in [*TOOLS.iterdir(),MONITOR,LOCAL/'closed.json',AMD/'failure-closed.json',PLATFORM/'closed.json',QUALIFIED/'closed.json'] if p.is_file()}
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
