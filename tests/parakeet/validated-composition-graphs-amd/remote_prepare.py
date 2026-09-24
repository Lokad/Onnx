"""Reuse exact reference tensors and the already qualified warmed consumer."""
from pathlib import Path
import copy,json,os,shutil,psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from checks import consumer_inventory
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
OLD=Path('/dev/shm/lokad-release-graph-baseline-v2-20260923')
WARM=Path('/dev/shm/lokad-warmed-release-v2-20260923')
BUILD=Path('/dev/shm/lokad-parakeet-validated-composition-models-20260924')
PRIOR=dict(baseline=OLD,warmed=WARM,build=BUILD,
    app=Path('/dev/shm/lokad-parakeet-validated-composition-app-20260924'),
    shared=Path('/dev/shm/lokad-parakeet-validated-composition-shared-20260924'),
    pyannote=Path('/dev/shm/lokad-parakeet-validated-composition-pyannote-20260924'))


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
    original_built=read(WARM/'built.json')
    assert pin(WARM/'built.json')==read(WARM/'collection.json')['files']['built.json']==pin(BASE/'evidence/warmed-consumer/built.json')
    review=consumer_inventory(read(BASE/'evidence/warmed-consumer/instructions.json'),read(WARM/'payload.json'),original_built)
    assert review==read(BASE/'evidence/warmed-consumer/review.json') and review['consumer']==stage['consumer']
    assert pin(BASE/'cases.json')==pin(OLD/'cases.json')==pin(WARM/'cases.json')
    shutil.copytree(OLD/'reference',BASE/'reference',copy_function=os.link)
    (BASE/'runtimes').mkdir();reused={}
    for role in ['current','candidate']:
        folder=BASE/'runtimes'/role;shutil.copytree(WARM/'runtimes/current',folder,copy_function=os.link)
        label='selected' if role=='current' else 'candidate'
        source=BUILD/'runtimes'/label/'Lokad.Onnx.dll'
        assert pin(source)==stage['products'][role]['Lokad.Onnx.dll']
        (folder/'Lokad.Onnx.dll').unlink();os.link(source,folder/'Lokad.Onnx.dll')
        assert pin(folder/'Lokad.Onnx.dll')==stage['products'][role]['Lokad.Onnx.dll']
        for name,wanted in stage['consumer_files'].items():
            assert original_built['files']['runtimes/current/'+name]==wanted==pin(folder/name)
            reused[(Path('runtimes')/role/name).as_posix()]=wanted
        cases=copy.deepcopy(read(BASE/'cases.json'));cases['core']=stage['products'][role]['Lokad.Onnx.dll']['sha256']
        save(BASE/('cases-'+role+'.json'),cases)
    built=dict(passed=True,reused_consumer=True,consumer=stage['consumer'],files=reused,
               original_receipt=pin(WARM/'built.json'),original_inventory=pin(BASE/'evidence/warmed-consumer/instructions.json'))
    save(BASE/'built.json',built)
    assert consumer_inventory(read(BASE/'evidence/warmed-consumer/instructions.json'),stage,built)==review
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        external=stage['external'],interpreter=stage['interpreter'],python_paths=stage['python_paths'],products=stage['products'],
        previous_consumer=stage['previous_consumer'],consumer=stage['consumer'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'})
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(payload['external']))))


if __name__=='__main__':main()
