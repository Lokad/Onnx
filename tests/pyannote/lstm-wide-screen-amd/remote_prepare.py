"""Copy unchanged current/candidate products and the full qualified AMD fixtures."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
FOLDERS=dict(screen=Path('/dev/shm/lokad-pyannote-lstm-input-screen-20260922'),codegen=Path('/dev/shm/lokad-pyannote-lstm-wide-codegen-20260923'),build=Path('/dev/shm/lokad-pyannote-lstm-wide-build-20260922'),focused=Path('/dev/shm/lokad-pyannote-lstm-wide-focused-20260923'),
    current=Path('/dev/shm/lokad-parakeet-current-baseline-20260922'),qualified=Path('/dev/shm/lokad-pyannote-lstm-input-blocks-v2-20260922'))


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for label,folder in FOLDERS.items():
        proof=read(BASE/'evidence'/(label+'-closed.json'));assert pin(folder/'collection.json')==proof['files']['collected/collection.json']
        receipt=read(folder/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        for identity in receipt['identities']:assert not live(identity)
        for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    qualified=FOLDERS['qualified'];old=read(qualified/'payload.json')
    assert pin(qualified/'payload.json')['sha256']=='9dc81ca197c65ede9d436d996d79268b4b3f496e66dbac72024e9614530d0bde'
    for name,wanted in old['files'].items():assert pin(qualified/name)==wanted,name
    for name in ['fixtures','references']:shutil.copytree(qualified/name,BASE/name)
    shutil.copytree(FOLDERS['screen']/'runtime/selected',BASE/'previous')
    assert pin(BASE/'previous/LstmScreen.dll')==stage['previous_consumer']
    for role,source in [('selected',FOLDERS['current']/'runtimes/current'),('candidate',FOLDERS['build']/'runtime')]:
        target=BASE/'runtime'/role;shutil.copytree(source,target)
        assert not list(target.glob('LstmScreen.*'))
        for name,wanted in stage['products'][role].items():assert pin(target/name)==wanted,name
    environment=read(FOLDERS['build']/'payload.json')
    for name,wanted in environment['external'].items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,schedule=read(BASE/'schedule.json'),limits=LIMITS,previous_owner=read(FOLDERS['codegen']/'collection.json')['identities'][0],boot_time=1789634288.0,
        products=stage['products'],cores={role:files['Lokad.Onnx.dll'] for role,files in stage['products'].items()},previous_consumer=stage['previous_consumer'],
        feed=environment['feed'],external=environment['external'],interpreter=environment['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Complete LSTM Reset+Execute screen; fixed current,candidate,candidate,current order, all twelve cases and every clock retained. Unchanged numerical and admission gates.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
