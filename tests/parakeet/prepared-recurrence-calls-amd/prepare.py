"""Freeze compiled products and reuse the closed actual-call capture without inference."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
import onnx
from onnx import numpy_helper
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-prepared-recurrence-calls-amd-20260924'
CAPTURE = ROOT/'artifacts/parakeet-decoder-lstm-capture-amd-20260924'
BUILD = ROOT/'artifacts/parakeet-prepared-recurrence-build-amd-20260924'
REVIEW = ROOT/'artifacts/parakeet-prepared-recurrence-build-review-20260924'
CONTRACTS = ROOT/'artifacts/parakeet-prepared-recurrence-contracts-amd-20260924'
SOURCE = ROOT/'artifacts/parakeet-prepared-recurrence-source-20260924'
RESIDENCY = ROOT/'artifacts/parakeet-inclusive-packing-residency-amd-v2-20260924'
PARENTS = [(CAPTURE,'28c7afe448ed16e3bb19d29232c3f90eb2d72c096196ae27792f5261afa5b64f',True),
           (BUILD,'b2d46d3b37e0df3d2df3828ab186760574feaaeb221c53f2346f8272dd5fdd6f',False),
           (REVIEW,'1b7f8e2130d490851e861790b2e051edfca3308be6d6d9e1067cda8b97f9869f',True),
           (CONTRACTS,'aacf2dbe9265a96c5699155a86b1743462387990245ca8fdc6cc0e75cc73952a',True),
           (RESIDENCY,'82342a58fddab63974e8bff009916a91376beb1d91f9db264480c2763c8e7b0a',True)]


def previous_closed():
    for folder,digest,passed in PARENTS:
        assert pin(folder/'closed.json')['sha256']==digest
        value=read(folder/'closed.json');assert value['passed'] is passed
        root=ROOT if value.get('paths_relative_to_repository') else folder
        for name,wanted in value['files'].items():assert pin(root/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='b52a89c1043165de1c376b37fc5307cd003a7c8b76f0f52508c4cdefcc669eab'
    for name,wanted in read(SOURCE/'prepared.json')['before'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['Calls.cs','Calls.csproj']:copy(TOOLS/name,bundle/'source'/name)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'prospective-calls.md')
    for label,folder in [('capture',CAPTURE),('build',BUILD),('contracts',CONTRACTS)]:
        for name in ['closed.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    for name in ['closed.json','analysis.json']:copy(REVIEW/name,bundle/'evidence/review'/name)
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(RESIDENCY/'closed.json',bundle/'evidence/residency-closed.json')
    copy(RESIDENCY/'collected/selected-decoder-512/result.json',bundle/'evidence/matrix-residency.json')
    copy(CAPTURE/'collected/capture-spec.json',bundle/'decoder-spec.json')
    identities={}
    for role,folder in [('selected',CAPTURE/'collected/runtime'),('candidate',BUILD/'collected/runtime')]:
        identities[role]={}
        for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Google.Protobuf.dll']:
            copy(folder/name,bundle/'products'/role/name);identities[role][name]=pin(folder/name)
    assert identities['selected']['Lokad.Onnx.dll']['sha256']=='672e5f303b011e27bb23097a49252c38ddee334938c75895e3c2341df0f3be35'
    assert identities['candidate']['Lokad.Onnx.dll']['sha256']=='3c23b44a53736938f241302a6a8258605c2f57da57f5009c83e979337df957f5'
    assert identities['candidate']['Lokad.Onnx.Data.dll']['sha256']=='cc37b19eb41cf728061c35bcdb7e06a6ab4d370bfd4c47555eec86c86b2ef6d6'
    decoder=read(bundle/'decoder-spec.json');links={}
    def link(source,name):
        wanted=pin(source);originals[source.relative_to(ROOT).as_posix()]=wanted
        links[name]=dict(source=source.relative_to(CAPTURE/'collected').as_posix(),pin=wanted)
    for source in (CAPTURE/'collected/capture/output').iterdir():
        if source.is_file():link(source,'fixtures/'+source.name)
    for source in (CAPTURE/'collected/native').iterdir():
        if source.is_file() and source.name!='review.json':link(source,'native/'+source.name)
    for case in decoder['cases']:
        if case['encoder']:
            name=case['encoder']['file'];link(CAPTURE/'collected/selected-arrays'/name,'selected-arrays/'+name)
    model=ROOT/'models/parakeet-tdt-0.6b-v3/decoder_joint-model.onnx';originals[model.relative_to(ROOT).as_posix()]=pin(model)
    assert pin(model)['sha256']==decoder['model_sha256'];graph=onnx.load(model,load_external_data=False)
    initializers={};panels=[];recurrent={name for node in decoder['nodes'] for name in node['input_names'][1:3]}
    for t in graph.graph.initializer:
        a=numpy_helper.to_array(t);assert a.dtype==np.dtype('<f4') and np.isfinite(a).all()
        meta=dict(shape=list(a.shape),bytes=a.nbytes,sha256=hashlib.sha256(a.tobytes()).hexdigest());initializers[t.name]=meta
        if t.name in recurrent:
            assert list(a.shape)==[1,2560,640]
            panels.append(dict(kind='recurrent',name=t.name,shape=list(a.shape),bytes=a.nbytes,source_sha256=meta['sha256'],
                prepared_sha256=hashlib.sha256(np.ascontiguousarray(a.transpose(0,2,1)).tobytes()).hexdigest()))
    assert len(initializers)==13 and len(panels)==4
    matrices=[dict(kind='matrix',name=w['name'],shape=w['shape'],bytes=w['bytes'],source_sha256=w['source_sha256'],prepared_sha256=w['packed_sha256'])
              for w in read(bundle/'evidence/matrix-residency.json')['weights']]
    assert len(matrices)==3 and sum(w['bytes'] for w in matrices)==25246720
    save(bundle/'spec.json',dict(identities=identities,initializers=initializers,recurrent=sorted(panels,key=lambda w:w['name']),matrices=matrices))
    save(bundle/'stage.json',dict(passed=True,identities=identities,links=links,model=pin(model),model_path=decoder['model'],
        onnx_parser_version=onnx.__version__,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz'),linked_files=len(links),decoder_executions=1520,complete_calls=3040)))


if __name__=='__main__':prepare()
