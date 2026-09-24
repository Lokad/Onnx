"""Freeze selected trajectories, byte-exact recurrent nodes and bounded capture inputs."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import onnx
from onnx import helper, numpy_helper, TensorProto
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-decoder-lstm-capture-amd-20260924'
MODELS = ROOT/'artifacts/parakeet-inclusive-packing-models-amd-20260924'
APP = ROOT/'artifacts/parakeet-inclusive-packing-app-amd-20260924'
BUILD = ROOT/'artifacts/parakeet-inclusive-packing-build-amd-20260924'
RELEASE = ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
SOURCE = ROOT/'artifacts/parakeet-inclusive-packing-source-20260924'
PARENTS = [(MODELS,'2d6aedeeaa8cdec4083a462f117272881c544f79f1464c6425e451f53fbfc66c'),
           (APP,'5a7344ce05b9d2d8c6cf8cd8c7a618030a594424fe160a645a088bdf96a600dc'),
           (BUILD,'50f2a3a8a2ebbe20d41315bc0be4242e24fb7f3eb74e78533091a750791a4078'),
           (RELEASE,'16d570819ab69915fe34fa6c5a4efb79d45ae792dbd5b1448c65645fa0d55f73')]
SHAPES = [[1,1,640],[1,2560,640],[1,2560,640],[1,5120],None,[1,1,640],[1,1,640]]
OUTPUT_SHAPES = [[1,1,1,640],[1,1,640],[1,1,640]]


def previous_closed():
    for folder,digest in PARENTS:
        assert pin(folder/'closed.json')['sha256']==digest
        value=read(folder/'closed.json');assert value['passed']
        for name,wanted in value['files'].items():assert pin(folder/name)==wanted,name
    assert not read(APP/'closed.json')['admitted']
    for name,wanted in read(SOURCE/'prepared.json')['before'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['Capture.cs','Capture.csproj']:copy(TOOLS/name,bundle/'source'/name)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','native.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'prospective-capture.md')
    for label,folder in [('models',MODELS),('app',APP),('build',BUILD)]:
        for name in ['closed.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    copy(RELEASE/'closed.json',bundle/'evidence/release-closed.json')
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    for name in ['result.json']:
        copy(MODELS/'collected/selected-native-512'/name,bundle/'evidence/selected-result.json')
    copy(MODELS/'collected/parakeet-reference/manifest.json',bundle/'evidence/native-manifest.json')
    identities={}
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Google.Protobuf.dll']:
        copy(MODELS/'collected/runtimes/selected'/name,bundle/'runtime'/name);identities[name]=pin(bundle/'runtime'/name)
    selected=read(bundle/'evidence/selected-result.json');reference=read(bundle/'evidence/native-manifest.json')
    assert selected['core_sha256']==identities['Lokad.Onnx.dll']['sha256']=='672e5f303b011e27bb23097a49252c38ddee334938c75895e3c2341df0f3be35'
    assert selected['data_sha256']==identities['Lokad.Onnx.Data.dll']['sha256']=='065b7a7f28561a37174f9d38ac13ef77bc59c010702e2b64200e9542376eb4c5'
    cases=[];arrays={}
    for row,case in zip(selected['rows'],reference['cases'],strict=True):
        assert row['name']==case['name'];steps=[]
        lookup={(v['label'],v['output']):v for v in row['comparisons']}
        def descriptor(label,output):
            item=lookup[label,output];file=MODELS/'collected/selected-native-512/result.json.tensors'/item['file']
            wanted=pin(file);assert wanted['sha256']==item['sha256']
            originals[file.relative_to(ROOT).as_posix()]=wanted;arrays[item['file']]=wanted
            return {k:item[k] for k in ['file','shape','dtype','sha256']}|{'bytes':wanted['bytes']}
        for i,step in enumerate(case['steps']):
            steps.append({k:step[k] for k in ['frame','target','token','duration']}|
                         {'outputs':{name:descriptor('step-'+str(i),name) for name in step['outputs']}})
        cases.append(dict(name=row['name'],steps=steps,encoder=None if not steps else descriptor('encoder','outputs')))
    assert [len(c['steps']) for c in cases]==[37,29,46,4,37,0,37]
    model=ROOT/'models/parakeet-tdt-0.6b-v3/decoder_joint-model.onnx';model_pin=pin(model)
    assert model_pin==reference['assets']['files']['decoder_joint-model.onnx'];originals[model.relative_to(ROOT).as_posix()]=model_pin
    graph=onnx.load(model,load_external_data=False);nodes=[n for n in graph.graph.node if n.op_type=='LSTM'];assert len(nodes)==2
    constants={t.name:t for t in graph.graph.initializer};derived=[];(bundle/'models').mkdir()
    for i,node in enumerate(nodes):
        assert len(node.input)==7 and node.input[4]=='' and len(node.output)==3
        assert {a.name:helper.get_attribute_value(a) for a in node.attribute}==dict(hidden_size=640)
        inputs=[helper.make_tensor_value_info(n,TensorProto.FLOAT,s) for n,s in zip(node.input,SHAPES) if n]
        outputs=[helper.make_tensor_value_info(n,TensorProto.FLOAT,s) for n,s in zip(node.output,OUTPUT_SHAPES)]
        one=helper.make_model(helper.make_graph([node],f'actual-decoder-lstm-{i}',inputs,outputs),
                              ir_version=graph.ir_version,opset_imports=graph.opset_import)
        onnx.checker.check_model(one);path=bundle/'models'/f'lstm-{i}.onnx';onnx.save(one,path)
        assert onnx.load(path).graph.node[0].SerializeToString()==node.SerializeToString()
        weights={}
        for name in node.input[1:4]:
            a=numpy_helper.to_array(constants[name]);assert a.dtype.name=='float32'
            weights[name]=dict(shape=list(a.shape),bytes=a.nbytes,sha256=hashlib.sha256(a.tobytes()).hexdigest())
        derived.append(dict(index=i,name=node.name,file=path.relative_to(bundle).as_posix(),model=pin(path),
            input_names=list(node.input),output_names=list(node.output),input_shapes=SHAPES,output_shapes=OUTPUT_SHAPES,
            attributes=dict(hidden_size=640),opset=17,weights=weights,node_sha256=hashlib.sha256(node.SerializeToString()).hexdigest()))
    capture=dict(core=identities['Lokad.Onnx.dll']['sha256'],data=identities['Lokad.Onnx.Data.dll']['sha256'],
        model='/home/vermorel/Onnx/models/parakeet-tdt-0.6b-v3/decoder_joint-model.onnx',model_sha256=model_pin['sha256'],
        selected_arrays='selected-arrays',cases=cases,nodes=derived)
    save(bundle/'capture-spec.json',capture)
    save(bundle/'stage.json',dict(passed=True,identities=identities,arrays=arrays,model=model_pin,model_path=capture['model'],
        onnx_parser_version=onnx.__version__,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz'),steps=190,calls=380)))


if __name__=='__main__':prepare()
