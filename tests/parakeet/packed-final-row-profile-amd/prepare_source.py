"""Prepare the existing phase observer on exact M78 Data, without a VM build."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
SOURCE=ROOT/'artifacts/parakeet-packed-final-row-source-20260925'
BUILD=ROOT/'artifacts/parakeet-packed-final-row-build-amd-20260925'
MODELS=ROOT/'artifacts/parakeet-packed-final-row-models-amd-20260925'
PHASE=ROOT/'artifacts/parakeet-managed-phase-amd-20260924'
CONSUMER=ROOT/'artifacts/parakeet-slice-materialization-profile-amd-20260924'
OUT=ROOT/'artifacts/parakeet-packed-final-row-profile-source-20260925'
PREFIX='src/Lokad.Onnx.Data/'
TRANSCRIBER=PREFIX+'ParakeetTranscriber.cs'
PREPARE=b'        encoder.PrepareOwnedMatMulWeights();'
ANCHOR=b'    static IReadOnlyDictionary<string, ITensor> Execute(GraphExecution context, Dictionary<string, ITensor> feeds)\n    {\n'
HOOK=b'        using var observation = ParakeetPhaseProbe.Enter(context);\n'


def read(path):return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def hook_parts(original):
    newline=b'\r\n' if b'\r\n' in original else b'\n'
    assert original.count(b'\n')==original.count(newline),'Mixed source line endings'
    anchor=ANCHOR.replace(b'\n',newline);hook=HOOK.replace(b'\n',newline)
    assert original.count(anchor)==1 and original.count(PREPARE)==1
    assert hook not in original
    return anchor,hook


def instrument(original):
    anchor,hook=hook_parts(original)
    return original.replace(anchor,anchor+hook,1)


def verify_hook(original,observed):
    anchor,hook=hook_parts(original)
    assert observed==original.replace(anchor,anchor+hook,1)
    assert observed.count(PREPARE)==observed.count(hook)==1
    assert observed.replace(hook,b'',1)==original
    return dict(passed=True,original_bytes_exact_after_hook_removal=True,
                owned_weight_preparation_preserved=True,hook_at_execute_entry=True)


def prerequisites():
    assert pin(SOURCE/'prepared.json')['sha256']=='e3ca64b50ea4a5dd276d90a19779d602f8ed78b6275b9e5448fe3b021bec5e1b'
    source=read(SOURCE/'prepared.json');assert source['passed']
    proof=read(BUILD/'closed.json');review=read(BUILD/'build-review.json')
    assert proof['passed'] and proof['compiled_review']==pin(BUILD/'build-review.json')
    assert review['passed'] and review['source']==pin(SOURCE/'prepared.json')
    model_proof=read(MODELS/'closed.json');models=read(MODELS/'analysis.json')
    assert model_proof['passed'] and model_proof['analysis']==pin(MODELS/'analysis.json')
    assert models['passed'] and models['identities']['candidate']==review['product']
    assert review['product']['Lokad.Onnx.dll']['sha256']=='49901366484570493b7a42028e5c5458f30fe2c1703a01335b68d9a5f9a1fea9'
    assert review['product']['Lokad.Onnx.Data.dll']['sha256']=='01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1'
    for name,wanted in review['product'].items():
        assert pin(BUILD/'build-collected/runtime'/name)==wanted
        assert proof['files']['build-collected/runtime/'+name]==wanted
    phase=read(PHASE/'closed.json');phase_review=read(PHASE/'build-review.json')
    assert phase['passed'] and phase['build_review']==pin(PHASE/'build-review.json')
    assert phase_review['passed'] and phase_review['core_unchanged']
    data_review,=[r for r in phase_review['methods'] if r['assembly']=='Lokad.Onnx.Data.dll']
    assert (data_review['original_methods'],data_review['unchanged'],data_review['added'])==(697,696,50)
    assert data_review['public_surface_equal'] and data_review['original_flags_equal']
    assert data_review['changed'].startswith('Lokad.Onnx.ParakeetTranscriber::Execute::')
    spec=read(PHASE/'bundle/spec.json')
    assert phase_review['spec']==pin(PHASE/'bundle/spec.json')
    for name in ['data-source/PhaseProbe.cs','data-source/ObserverData.csproj',
                 'bridge-source/Program.cs','bridge-source/Bridge.csproj']:
        assert pin(PHASE/'bundle'/name)==spec['files'][name]
    consumer_proof=read(CONSUMER/'closed.json');consumer_review=read(CONSUMER/'build-review.json')
    assert consumer_proof['passed'] and consumer_proof['build_review']==pin(CONSUMER/'build-review.json')
    assert consumer_review['passed'] and consumer_review['products_byte_exact']
    assert (consumer_review['consumer']['original_methods'],consumer_review['consumer']['unchanged'])==(164,163)
    assert consumer_review['consumer']['flags_equal'] and consumer_review['consumer']['public_surface_equal']
    built=read(CONSUMER/'build-collected/built.json')
    assert consumer_review['built']==pin(CONSUMER/'build-collected/built.json')
    assert built['consumer']['sha256']=='38ab5c7e65aa00ace50e8704348830286047e0c96ebc0a04b66c6e54d063899c'
    return source,review,built


def prepare():
    assert not OUT.exists()
    source,review,built=prerequisites()
    data={n:v for n,v in source['source'].items() if n.startswith(PREFIX) and n.endswith('.cs')}
    assert len(data)==22 and TRANSCRIBER in data
    for name,wanted in data.items():assert pin(SOURCE/'source'/name)==wanted,name
    original=(SOURCE/'source'/TRANSCRIBER).read_bytes()
    observed=instrument(original);scope=verify_hook(original,observed)
    # Establish why the old diagnostic Data cannot be reused. All other Data
    # source text agrees; its transcriber lacks exactly the owned-preparation call.
    for name in data:
        old=(PHASE/'bundle/data-source'/Path(name).name).read_text(encoding='utf8')
        new=(SOURCE/'source'/name).read_text(encoding='utf8')
        if name==TRANSCRIBER:
            old=old.replace(HOOK.decode(),'',1)
            assert new.count(PREPARE.decode()+'\n')==1
            new=new.replace(PREPARE.decode()+'\n','',1)
        assert old==new,name
    OUT.mkdir();originals={}
    def put(name,content):
        path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(content)
    def copy(source,name):
        put(name,source.read_bytes());originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in data:
        path=SOURCE/'source'/name;target='data-source/'+Path(name).name
        if name==TRANSCRIBER:
            put(target,observed);originals[path.relative_to(ROOT).as_posix()]=pin(path)
        else:copy(path,target)
    for name in ['data-source/PhaseProbe.cs','data-source/ObserverData.csproj',
                 'bridge-source/Program.cs','bridge-source/Bridge.csproj']:
        copy(PHASE/'bundle'/name,name)
    assert pin(SOURCE/'source/global.json')==source['source']['global.json']
    for folder in ['data-source','bridge-source']:copy(SOURCE/'source/global.json',folder+'/global.json')
    runtime=['SampledAudio.dll','SampledAudio.deps.json','SampledAudio.runtimeconfig.json',
             'FastBertTokenizer.dll','Google.Protobuf.dll','Lokad.Tokenizers.dll','SixLabors.ImageSharp.dll']
    for name in runtime:
        path=CONSUMER/'build-collected/runtime-observed'/name
        assert pin(path)==built['runtime_files']['runtime-observed/'+name]
        copy(path,'runtime-base/'+name)
    for name,wanted in review['product'].items():copy(BUILD/'build-collected/runtime'/name,'runtime-base/'+name)
    for label,folder in [('product',BUILD),('models',MODELS),('phase',PHASE),('consumer',CONSUMER)]:
        copy(folder/'closed.json','evidence/'+label+'/closed.json')
        if label!='models':copy(folder/'build-review.json','evidence/'+label+'/build-review.json')
        else:copy(folder/'analysis.json','evidence/'+label+'/analysis.json')
    copy(SOURCE/'prepared.json','evidence/source-prepared.json')
    copy(TOOLS/'README.md','prospective-observation.md')
    for name in ['prepare_source.py','test_source.py']:originals[(TOOLS/name).relative_to(ROOT).as_posix()]=pin(TOOLS/name)
    receipt=dict(passed=True,product=review['product'],consumer=built['consumer'],
        source=pin(SOURCE/'prepared.json'),data_sources=data,scope=scope,
        previous_observer_cannot_be_reused=True,previous_observer_missing_owned_preparation=True,
        phase_probe_unchanged=True,request_consumer_rebuilt=False,core_rebuilt=False,
        built=False,root_product_changed=False,release_admitted=False,
        compiled_review_required=True,request_and_graph_profile_qualification_required=True,
        files={p.relative_to(OUT).as_posix():pin(p) for p in OUT.rglob('*') if p.is_file()},inputs=originals)
    with (OUT/'prepared.json').open('x',encoding='utf8') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=True,prepared=pin(OUT/'prepared.json'),source_files=len(data),
        product=review['product'],consumer=built['consumer'],scope=scope,built=False)))


if __name__=='__main__':prepare()
