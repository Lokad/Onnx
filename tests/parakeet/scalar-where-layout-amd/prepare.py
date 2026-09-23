"""Freeze selected encoder inputs and references without copying any model."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-scalar-where-layout-amd-20260923'
BUILD = ROOT/'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
MODELS = ROOT/'artifacts/parakeet-wide-entry-first-use-models-amd-20260923'
RELEASE = ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
PRIOR = ROOT/'artifacts/parakeet-ordered-wide-blocks-screen-amd-20260923'
CASES = ['english-16k', 'jfk-48k-stereo']
SOURCE_FILES = ['src/Lokad.Onnx/ComputationalGraph.cs','src/Lokad.Onnx/GraphExecution.cs',
    'src/Lokad.Onnx/Log.cs','src/Lokad.Onnx/TensorOps.Elementwise.cs','src/Lokad.Onnx/BroadcastedTensor.cs']


def previous_closed():
    for folder,digest in [(BUILD,'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'),
        (MODELS,'f30100534cbb79db790aac30365d533e6d3dcce79e779abc5feb8f7cf3fc1e22'),
        (RELEASE,'16d570819ab69915fe34fa6c5a4efb79d45ae792dbd5b1448c65645fa0d55f73'),
        (PRIOR,'9656e7102e7b510877c915dd37117591f8a9e2d3a9c32fcc173e83ae00780ee0')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert not read(PRIOR/'closed.json')['admitted']
    product=read(BUILD/'analysis.json')['built']
    assert read(RELEASE/'analysis.json')['root_source_verified']
    assert read(RELEASE/'analysis.json')['measured']==product
    source=read(RELEASE/'bundle/evidence/root-applied.json')['source_files']
    assert len(source)==422
    for name,wanted in source.items():assert pin(ROOT/name)==wanted,name
    result=read(MODELS/'collected/candidate-native/result.json')
    assert result['passed'] and result['core_sha256']==product['Lokad.Onnx.dll']['sha256']
    assert result['data_sha256']==product['Lokad.Onnx.Data.dll']['sha256']
    return product


def prepare():
    assert not BASE.exists();product=previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['Program.cs','Prototype.csproj']:copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m56-parakeet-scalar-where-20260923.md',bundle/'prospective-plan.md')
    for label,folder in [('build',BUILD),('models',MODELS),('release',RELEASE),('prior',PRIOR)]:
        for name in ['closed.json','payload.json']:copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    copy(MODELS/'collected/candidate-native/result.json',bundle/'evidence/selected-native.json')
    native=MODELS/'collected/parakeet-reference'
    copy(native/'manifest.json',bundle/'evidence/native-manifest.json')
    rows={r['name']:r for r in read(MODELS/'collected/candidate-native/result.json')['rows']}
    native_manifest=read(native/'manifest.json');native_cases={r['name']:r for r in native_manifest['cases']}
    cases=[]
    for name in CASES:
        comparisons={(r['label'],r['output']):r for r in rows[name]['comparisons']}
        inputs={};outputs={};native_outputs={}
        for label,keys,target in [('frontend',[('features','audio_signal'),('features_lens','length')],inputs),
                                  ('encoder',[('outputs','outputs'),('encoded_lengths','encoded_lengths')],outputs)]:
            for original,key in keys:
                row=comparisons[label,original]
                path=MODELS/'collected/candidate-native/result.json.tensors'/row['file']
                assert pin(path)['sha256']==row['sha256']
                dest=bundle/'inputs'/row['file'];copy(path,dest)
                target[key]=dict(file=dest.relative_to(bundle).as_posix(),shape=row['shape'],dtype=row['dtype'],**pin(dest))
        for key,file in native_cases[name]['stages'][1]['outputs'].items():
            assert pin(native/file)=={k:native_manifest['files'][file][k] for k in ['bytes','sha256']}
            dest=bundle/'native'/file;copy(native/file,dest)
            native_outputs[key]=dict(file=dest.relative_to(bundle).as_posix(),**native_manifest['files'][file])
        cases.append(dict(name=name,inputs=inputs,outputs=outputs,native_outputs=native_outputs))
    observations=ROOT/'tests/parakeet/current-profile-results/memory-source-observations-20260923.json'
    copy(observations,bundle/'evidence/graph-observations.json')
    models={}
    for name in ['encoder-model.onnx','encoder-model.onnx.data']:
        expected=native_manifest['assets']['files'][name];assert pin(ROOT/'models/parakeet-tdt-0.6b-v3'/name)==expected
        models['/home/vermorel/Onnx/models/parakeet-tdt-0.6b-v3/'+name]=expected
    save(bundle/'manifest.json',dict(core_sha256=product['Lokad.Onnx.dll']['sha256'],
        encoder='/home/vermorel/Onnx/models/parakeet-tdt-0.6b-v3/encoder-model.onnx',cases=cases,
        where_nodes=73,fixture_names=['/Where','/layers.0/self_attn/Where','/layers.0/self_attn/Where_1','/layers.0/conv/Where'],
        fixture_count=8,export_cap_bytes=32*1024**2,scope='Two selected encoder requests, logger metadata and eight fixed fixtures; no timing.'))
    for p in (BUILD/'collected/runtime').iterdir():
        if p.is_file():copy(p,bundle/'runtimes/current'/p.name)
    for name in SOURCE_FILES:copy(ROOT/name,bundle/'evidence/source'/Path(name).name)
    save(bundle/'stage.json',dict(passed=True,product=product,models=models,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),product=product,cases=CASES)))


if __name__=='__main__':prepare()
