"""Link exact terminal inputs and reserve fresh consumer outputs."""
import copy,json,os
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]


def main():
    psutil.Process().cpu_affinity([0]);idle();assert psutil.boot_time()==1789634288.0
    assert not (BASE/'payload.json').exists()
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for label,remote in [('graph','/dev/shm/lokad-parakeet-validated-composition-graphs-20260924'),
                         ('diagnostic','/dev/shm/lokad-e5-runtime-diagnostic-20260924')]:
        receipt=read(BASE/'evidence'/label/'collection.json')
        assert pin(Path(remote)/'collection.json')==pin(BASE/'evidence'/label/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
    for name,link in stage['links'].items():
        target=(BASE/name).resolve();assert target.is_relative_to(BASE.resolve()) and not target.exists()
        source=Path(link['source']);assert pin(source)==link['identity'],str(source)
        target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
    assert pin(BASE/'previous/ReleaseBenchmark.dll')==stage['previous_consumer']
    for role in ['current','candidate']:
        assert pin(BASE/'runtimes'/role/'Lokad.Onnx.dll')==stage['products'][role]['Lokad.Onnx.dll']
        cases=copy.deepcopy(read(BASE/'cases.json'));cases['core']=stage['products'][role]['Lokad.Onnx.dll']['sha256']
        save(BASE/f'cases-{role}.json',cases)
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        feed=stage['feed'],external=stage['external'],interpreter=stage['interpreter'],python_paths=stage['python_paths'],
        products=stage['products'],previous_consumer=stage['previous_consumer'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,files=len(payload['files']),external=len(payload['external']))))


if __name__=='__main__':main()
