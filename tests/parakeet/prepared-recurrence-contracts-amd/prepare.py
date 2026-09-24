"""Freeze new lifecycle tests and existing LSTM/packing contracts against the actual DLLs."""
import ast
import json
from pathlib import Path
import re
import shutil
import tarfile
from protocol import pin, read, save

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-contracts-amd-20260924'
BUILD=ROOT/'artifacts/parakeet-prepared-recurrence-build-amd-20260924'
REVIEW=ROOT/'artifacts/parakeet-prepared-recurrence-build-review-20260924'
SOURCE=ROOT/'artifacts/parakeet-prepared-recurrence-source-20260924'
PARENT=ROOT/'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
TESTS=['PreparedLstmWeightsTests.cs','CpuExecutionProviderLstmTests.cs','LstmInputBlockTests.cs','LstmPanelAdmissionTests.cs',
       'LstmPanelOverflowRefusalTests.cs','LstmOutputLaneTests.cs','FoldBudgetTests.cs','PackedWeightsTests.cs','PackedWeightBudgetTests.cs']


def previous_closed():
    assert pin(REVIEW/'closed.json')['sha256']=='1b7f8e2130d490851e861790b2e051edfca3308be6d6d9e1067cda8b97f9869f'
    proof=read(REVIEW/'closed.json');assert proof['passed'] and proof['paths_relative_to_repository']
    for name,wanted in proof['files'].items():assert pin(ROOT/name)==wanted,name
    assert pin(BUILD/'closed.json')['sha256']=='b2d46d3b37e0df3d2df3828ab186760574feaaeb221c53f2346f8272dd5fdd6f'
    refusal=read(BUILD/'closed.json');assert not refusal['passed'] and refusal['build_jobs_passed'] and refusal['inference_calls']==0
    for name,wanted in refusal['files'].items():assert pin(BUILD/name)==wanted,name
    assert pin(PARENT/'closed.json')['sha256']=='da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'
    proof=read(PARENT/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(PARENT/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='b52a89c1043165de1c376b37fc5307cd003a7c8b76f0f52508c4cdefcc669eab'
    source=read(SOURCE/'prepared.json')
    for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={};expected={};census={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in TESTS+['IdentityTests.cs','PreparedBudgetReceiptTests.cs']:
        path=(SOURCE/'source/tests/Lokad.Onnx.Backend.Tests' if name in TESTS else TOOLS)/name
        copy(path,bundle/'source'/name)
        text=path.read_text();assert not any(attribute in text for attribute in ['[MemberData','[ClassData','[Skippable'])
        found=re.findall(r'((?:\s*\[(?:Fact|Theory|InlineData\([^\n]*\))\]\s*)+)public void (\w+)',text)
        methods={'Lokad.Onnx.Backend.Tests.'+name[:-3]+'.'+method:attributes.count('[InlineData(') or 1 for attributes,method in found}
        assert methods and len(methods)==len(found);expected.update(methods);census[name]=sum(methods.values())
    assert census['PreparedLstmWeightsTests.cs']==22 and census['IdentityTests.cs']==census['PreparedBudgetReceiptTests.cs']==1
    copy(SOURCE/'source/global.json',bundle/'source/global.json');copy(TOOLS/'PackingContracts.csproj',bundle/'source/PackingContracts.csproj')
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(TOOLS/name,bundle/'tools'/name)
    identities={}
    for role,folder in [('selected',PARENT),('candidate',BUILD)]:
        for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Google.Protobuf.dll']:copy(folder/'collected/runtime'/name,bundle/'products'/role/name)
        identities[role]={name:pin(bundle/'products'/role/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
        assert identities[role]==read((REVIEW if role=='candidate' else PARENT)/'analysis.json')['built']
    for name in ['closed.json','payload.json']:copy(BUILD/name,bundle/'evidence'/name)
    for name in ['closed.json','analysis.json']:copy(REVIEW/name,bundle/'evidence'/('review-'+name))
    copy(BUILD/'collected/collection.json',bundle/'evidence/collection.json')
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json');copy(TOOLS/'README.md',bundle/'prospective-contracts.md')
    save(bundle/'stage.json',dict(passed=True,identities=identities,expected_cases=expected,census=census,
        source_prepared=pin(SOURCE/'prepared.json'),parent_build=pin(REVIEW/'closed.json'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz'),tests=sum(expected.values()),census=census)))


if __name__=='__main__':prepare()
