"""Freeze normal candidate source and independent package consumption, without local builds."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-input-product-amd-20260922'
CANDIDATE=ROOT/'artifacts/pyannote-lstm-input-blocks-v2-20260922'
SCREEN=ROOT/'artifacts/pyannote-lstm-input-screen-amd-20260922'
PACKAGE=ROOT/'artifacts/pyannote-blocked-spatial-package-20260922'
APP=ROOT/'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922'
ROOT_PROOF=ROOT/'artifacts/pyannote-blocked-spatial-root-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('product_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    assert pin(SCREEN/'closed.json')['sha256']=='ed5edcaf7ac71c89acb57e3d82ecf80c6239d4aa46210a02e0f8b72adc71ccd5'
    for folder in [SCREEN,PACKAGE]:
        for name,wanted in read(folder/'closed.json')['files'].items():assert pin(folder/name)==wanted,name
    assert read(SCREEN/'analysis.json')['decision']['admitted']
    assert pin(PACKAGE/'closed.json')['sha256']=='fc4f4811032ff38ea837b8d53ea68325a63cd9959b7e5d61b9fbb89ba7fefc66'
    assert pin(CANDIDATE/'failure-closed.json')['sha256']=='f375ecbf942d47e429f0fad51dbd02e32ed59d5800a27e904dcd13786518acde'
    candidate=read(CANDIDATE/'failure-closed.json')
    for name in ['inputs.json','binaries.json','test-results/lstm-ordinary.trx']:
        assert pin(CANDIDATE/name)==candidate['files'][name]
    monitor.verify(read(CANDIDATE/'inputs.json')['files'])
    monitor.verify(read(CANDIDATE/'binaries.json')['files'])
    assert pin(APP/'closed.json')['sha256']=='5c238cd33845eb58fc00332361a967185ae82a9fcb70854530e07fad58f064d0'
    assert pin(ROOT_PROOF/'closed.json')['sha256']=='301fe7a2291f4302aa7b4d24622e72bfd723af129920c753cb86819b9051018f'


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for key,wanted in read(CANDIDATE/'inputs.json')['files'].items():
        path=Path(key)
        if path.is_relative_to(CANDIDATE/'source'):
            relative=path.relative_to(CANDIDATE/'source');assert not {'bin','obj'}.intersection(relative.parts)
            assert pin(path)==wanted;copy(path,bundle/'source'/relative)
    (bundle/'evidence').mkdir()
    archive=bundle/'evidence/tensor-source.tar'
    subprocess.run(['git','archive','--format=tar','--output',str(archive),'9533cd67','--','tests/Lokad.Onnx.Tensors.Tests','Lokad.Onnx.slnx'],cwd=ROOT,check=True)
    with tarfile.open(archive) as tar:
        assert all((m.name.startswith('tests/') or m.name=='Lokad.Onnx.slnx') and '..' not in Path(m.name).parts for m in tar.getmembers())
        tar.extractall(bundle/'source',filter='data')
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(TOOLS/name,bundle/'tools'/name)
    for name in ['Program.cs','PackageProbe.csproj']:copy(PACKAGE/'consumer'/name,bundle/'consumer'/name)
    for name in ['Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json']:
        source=ROOT_PROOF/'bridge/bin/Release/net10.0'/name
        assert read(ROOT_PROOF/'closed.json')['files'][source.relative_to(ROOT).as_posix()]==pin(source)
        copy(source,bundle/'bridge'/name)
    for name in ['backend.trx','tensors.trx']:
        source=APP/'collected/campaign/test-results'/name
        assert pin(source)==read(APP/'closed.json')['files'][source.relative_to(APP).as_posix()]
        copy(source,bundle/'evidence'/('selected-'+name))
    copy(CANDIDATE/'test-results/lstm-ordinary.trx',bundle/'evidence/candidate-lstm.trx')
    copy(ROOT/'PLAN.md',bundle/'prospective-plan.md');originals.pop('PLAN.md')
    copy(ROOT/'artifacts/pyannote-lstm-product-space-20260922/receipt.json',bundle/'evidence/space-release.json')
    measured=read(SCREEN/'analysis.json')
    stage=dict(passed=True,measured_core=measured['cores']['candidate'],source_commit='9533cd67 plus qualified M22 changes',
        screen=pin(SCREEN/'closed.json'),tensor_source_commit=subprocess.check_output(['git','rev-parse','9533cd67'],cwd=ROOT,text=True).strip(),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    files=dict(originals)
    for p in [*TOOLS.iterdir(),MONITOR,SCREEN/'closed.json',PACKAGE/'closed.json',CANDIDATE/'inputs.json',CANDIDATE/'binaries.json',APP/'closed.json',ROOT_PROOF/'closed.json']:
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),source_files=sum(k.startswith('source/') for k in stage['files']))))


if __name__=='__main__':prepare()
