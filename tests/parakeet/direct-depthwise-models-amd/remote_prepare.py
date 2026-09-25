"""Bind unchanged consumers and model truth after direct-depthwise qualification."""
import copy
import json
import os
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
CURRENT=Path('/dev/shm/lokad-parakeet-packed-final-row-models-20260925')
PREVIOUS=Path('/dev/shm/lokad-pyannote-winograd-product-parakeet-20260923')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for folder,label in [(CURRENT,'current'),(PREVIOUS,'previous')]:
        assert pin(folder/'collection.json')==pin(BASE/'evidence'/(label+'-collection.json'))
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        assert pin(folder/'payload.json')==pin(BASE/'evidence'/(label+'-payload.json'))
        for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name
    previous_owner=None
    for terminal in stage['terminal_sources']:
        folder=Path(terminal['remote']);label=terminal['label'];kind=terminal['kind']
        for name in [kind+'-state.json',kind+'-collection.json']:
            assert pin(folder/name)==pin(BASE/'evidence'/(label+'-'+name))
        state=read(folder/(kind+'-state.json'));receipt=read(folder/(kind+'-collection.json'))
        assert state['complete'] and receipt['terminal'] and state['code']==receipt['code']==0
        ids=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
        assert not any(live(i) for i in ids)
        if label=='census':previous_owner=state['supervisor']
    for label in ['contracts','census']:
        closure=read(BASE/'evidence'/(label+'-closed.json'));analysis=read(BASE/'evidence'/(label+'-analysis.json'))
        assert closure['passed'] and analysis['passed'] and closure['analysis']==pin(BASE/'evidence'/(label+'-analysis.json'))
    contracts=read(BASE/'evidence/contracts-analysis.json');census=read(BASE/'evidence/census-analysis.json')
    build=read(BASE/'evidence/build-review.json');binding=read(BASE/'evidence/census-build-review.json')
    observed_spec=read(BASE/'evidence/observer-spec.json')
    assert build['passed'] and binding['passed']
    assert contracts['compiled_review']==pin(BASE/'evidence/build-review.json')
    assert contracts['product']==build['product']==observed_spec['before_product']==stage['identities']['candidate']
    assert census['products']==binding['products'] and census['source']==observed_spec['source']
    assert build['data_binary_unchanged'] and build['public_surface_unchanged'] and build['zero_added_warnings']
    assert [(s['mode'],s['passed'],s['skipped']) for s in contracts['suites']]==[('normal',8,0),('scalar',8,0)]
    assert census['exact_public_results'] and census['public_requests']==80
    assert census['observed']['every_geometry_exact'] and census['observed']['zero_generic_work']
    assert census['observed']['per_corpus']['direct_batches']==520 and not stage['release_admitted']
    original=read(PREVIOUS/'payload.json')
    for name,wanted in original['external'].items():assert pin(name)==wanted,name
    for name in ['assets','parakeet-reference']:shutil.copytree(PREVIOUS/name,BASE/name,copy_function=os.link)
    assert pin(BASE/'parakeet-reference/manifest.json')['sha256']=='3bad7d262b8809b1265c84c8e66d02ee38e7d4cff2d92014448976a9e161103c'
    assert pin(BASE/'evidence/original-manifest.json')==pin(CURRENT/'manifests/candidate-parakeet.json')
    (BASE/'manifests').mkdir();(BASE/'runtimes').mkdir()
    for role in ['selected','candidate']:
        folder=BASE/'runtimes'/role;shutil.copytree(CURRENT/'runtimes/candidate',folder,copy_function=os.link)
        for name,wanted in stage['identities'][role].items():
            assert pin(BASE/'products'/role/name)==wanted
            (folder/name).unlink();os.link(BASE/'products'/role/name,folder/name)
        for name,wanted in stage['consumers'].items():assert pin(folder/(name+'.dll'))==wanted
        manifest=copy.deepcopy(read(BASE/'evidence/original-manifest.json'))
        manifest.update(core_sha256=stage['identities'][role]['Lokad.Onnx.dll']['sha256'],data_sha256=stage['identities'][role]['Lokad.Onnx.Data.dll']['sha256'])
        manifest['product_source']='Direct depthwise selected M78' if role=='selected' else 'Direct nine-tap depthwise'
        save(BASE/'manifests'/(role+'-parakeet.json'),manifest)
    assert previous_owner is not None
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,boot_time=1789634288.0,previous_owner=previous_owner,
        identities=stage['identities'],consumers=stage['consumers'],external=original['external'],interpreter=original['interpreter'],
        failed_release_controls=stage['failed_release_controls'],release_admitted=False,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='784 arrays against pinned ORT reference and twenty full public clips per product/instruction mode. Exact selected bits and unchanged native 1e-4 gate. No scored performance.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
