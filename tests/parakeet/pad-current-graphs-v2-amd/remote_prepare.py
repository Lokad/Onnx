"""Link immutable references and three existing consumers to the actual products."""
import copy
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
from reuse import review

BASE=Path(__file__).resolve().parents[1]


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for label,remote in stage['prerequisites'].items():
        folder=Path(remote);evidence=BASE/'evidence/prerequisites'/label
        for name in ['payload.json','collection.json']:assert pin(folder/name)==pin(evidence/name)
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
    for name,wanted in stage['external'].items():assert pin(name)==wanted,name
    for name,row in stage['links'].items():
        target=BASE/name;assert target.resolve().is_relative_to(BASE.resolve()) and not target.exists()
        original=Path(row['source']);assert pin(original)==row['identity'],name
        target.parent.mkdir(parents=True,exist_ok=True);os.link(original,target)
    assert (BASE/'source/global.json').is_file(), 'The unchanged worker needs this working directory'
    failure=BASE/'evidence/launch-failure/collection.json'
    assert pin(failure)==pin(Path('/dev/shm/lokad-parakeet-pad-current-graphs-20260926/collection.json'))
    assert read(failure)['terminal'] and read(failure)['code']==1
    assert not any(live(i) for i in read(failure)['identities'])
    for role in ['current','candidate']:
        cases=copy.deepcopy(read(BASE/'cases.json'));cases['core']=stage['products'][role]['Lokad.Onnx.dll']['sha256']
        save(BASE/('cases-'+role+'.json'),cases)
    save(BASE/'consumer-reuse.json',review(BASE,stage,read(BASE/'built.json')))
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=stage['previous_owner'],boot_time=1789634288.0,
        **{name:stage[name] for name in ['external','interpreter','python_paths','products','previous_consumer','consumer','e5_consumer','short_consumer']},
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'})
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
