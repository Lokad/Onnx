"""Build the full test assembly against unchanged M26 product references."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-gates-focused-amd-20260923'
SOURCE=ROOT/'artifacts/pyannote-lstm-default-gates-20260923'
BUILD=ROOT/'artifacts/pyannote-lstm-gates-build-amd-20260923'
ORDINARY=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-20260922'
SCALAR=ROOT/'artifacts/pyannote-lstm-input-blocks-v3-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('wide_focused_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
PROJECT='source/tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'


def previous_closed():
    for folder,name,digest in [(BUILD,'closed.json','01af7cdacb5b0f0582b6b6e3739f12badfc32f02c39448715eb5804d850e33c9'),
            (SOURCE,'prepared.json','d836e93c0eae60b5c8729685d2097b0257320cdac1f1c536777938dbcbd5ef63'),
            (ORDINARY,'failure-closed.json','32a258cb1da52f0499e022dd3d0aa787db496fd1affe5c43c68a22093ffb9b13'),
            (SCALAR,'failure-closed.json','11c93f440a2be617083708421bd3af1239e91316acd04f059164514dfb40fd99')]:
        assert pin(folder/name)['sha256']==digest
        for key,wanted in read(folder/name)['files'].items():assert pin(folder/key)==wanted,key
    for key,wanted in read(SOURCE/'prepared.json')['before'].items():assert pin(ROOT/key)==wanted,key


def project_text(original):
    old='''    <ProjectReference Include="..\\..\\src\\Lokad.Onnx\\Lokad.Onnx.csproj" />
    <ProjectReference Include="..\\..\\src\\Lokad.Onnx.Data\\Lokad.Onnx.Data.csproj" />'''
    refs=['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']
    new='\n'.join('    <Reference Include="'+name+'"><HintPath>../../../runtime/'+name+'.dll</HintPath></Reference>' for name in refs)
    assert original.count(old)==1
    return original.replace(old,new)


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    before=read(SOURCE/'prepared.json')
    for name in before['files']:
        if name.startswith('source/'):copy(SOURCE/name,bundle/name)
    assert len(list((bundle/'source').rglob('*')))>418
    original=(bundle/PROJECT).read_text();(bundle/PROJECT).write_text(project_text(original),encoding='utf8',newline='\n')
    for name in before['files']:
        if name.startswith('source/') and name!=PROJECT:assert pin(bundle/name)==before['files'][name]
    copy(ORDINARY/'collected/suite-512/suite.trx',bundle/'references/ordinary.trx')
    copy(SCALAR/'test-results/lstm-scalar.trx',bundle/'references/scalar.trx')
    for folder,name in [(BUILD,'build'),(ORDINARY,'ordinary'),(SCALAR,'scalar')]:
        file=folder/('closed.json' if folder==BUILD else 'failure-closed.json');copy(file,bundle/'evidence'/(name+'-closed.json'))
    copy(BUILD/'analysis.json',bundle/'evidence/build-analysis.json');copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    for name in ['remote.py','remote_prepare.py','protocol.py','checks.py']:copy(TOOLS/name,bundle/'tools'/name)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    stage=dict(passed=True,product=read(BUILD/'analysis.json')['built'],files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),product=stage['product'])))


if __name__=='__main__':prepare()
