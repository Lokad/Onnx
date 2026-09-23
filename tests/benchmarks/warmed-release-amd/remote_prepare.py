"""Reuse exact model/fixture bytes while reserving new consumer destinations."""
from pathlib import Path
import copy,json,os,shutil,psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
BASE=Path(__file__).resolve().parents[1]
OLD=Path('/dev/shm/lokad-release-graph-baseline-v2-20260923')
BUILD=Path('/dev/shm/lokad-parakeet-first-use-kernels-build-20260923')
PRIOR=dict(baseline=OLD,build=BUILD,app=Path('/dev/shm/lokad-parakeet-first-use-kernels-app-20260923'),
    shared=Path('/dev/shm/lokad-parakeet-first-use-kernels-shared-20260923'),
    pyannote=Path('/dev/shm/lokad-parakeet-first-use-kernels-pyannote-20260923'),
    graph=Path('/dev/shm/lokad-parakeet-first-use-kernels-graphs-20260923'),
    diagnostic=Path('/dev/shm/lokad-graph-startup-diagnostic-20260923'))

def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for label,folder in PRIOR.items():
        for name in ['collection.json','payload.json']:assert pin(folder/name)==pin(BASE/'evidence'/label/name)
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name
    for name,wanted in stage['external'].items():assert pin(name)==wanted,name
    built=read(OLD/'built.json');assert pin(OLD/'built.json')==read(OLD/'collection.json')['files']['built.json']
    for name,wanted in built['files'].items():assert pin(OLD/name)==wanted,name
    assert pin(BASE/'cases.json')==pin(OLD/'cases.json')
    shutil.copytree(OLD/'reference',BASE/'reference',copy_function=os.link)
    shutil.copytree(OLD/'runtime',BASE/'previous',copy_function=os.link)
    assert pin(BASE/'previous/ReleaseBenchmark.dll')==stage['previous_consumer']
    (BASE/'product').mkdir()
    for name in ['Lokad.Onnx.dll','Google.Protobuf.dll']:os.link(OLD/'runtime'/name,BASE/'product'/name)
    (BASE/'runtimes').mkdir()
    for role in ['current','candidate']:
        folder=BASE/'runtimes'/role;folder.mkdir()
        for p in (OLD/'runtime').iterdir():
            if p.is_file() and not p.name.startswith('ReleaseBenchmark'):os.link(p,folder/p.name)
        if role=='candidate':
            (folder/'Lokad.Onnx.dll').unlink();os.link(BUILD/'runtime/Lokad.Onnx.dll',folder/'Lokad.Onnx.dll')
        assert pin(folder/'Lokad.Onnx.dll')==stage['products'][role]['Lokad.Onnx.dll']
        cases=copy.deepcopy(read(BASE/'cases.json'));cases['core']=stage['products'][role]['Lokad.Onnx.dll']['sha256']
        save(BASE/('cases-'+role+'.json'),cases)
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        external=stage['external'],interpreter=stage['interpreter'],python_paths=stage['python_paths'],products=stage['products'],
        previous_consumer=stage['previous_consumer'],feed=stage['feed'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'})
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(payload['external']))))

if __name__=='__main__':main()
