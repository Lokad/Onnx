"""Reuse byte-identical qualified consumer and complete selected-platform fixtures."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
FOLDERS=dict(numerics=Path('/dev/shm/lokad-pyannote-lstm-gates-replay-20260923'),build=Path('/dev/shm/lokad-pyannote-lstm-gates-build-20260923'),
    focused=Path('/dev/shm/lokad-pyannote-lstm-gates-focused-20260923'),
    current=Path('/dev/shm/lokad-parakeet-current-baseline-20260922'),
    qualified=Path('/dev/shm/lokad-pyannote-lstm-input-blocks-v2-20260922'),
    replay=Path('/dev/shm/lokad-pyannote-lstm-wide-codegen-20260923'))

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
    for role,source in [('selected',FOLDERS['current']/'runtimes/current'),('candidate',FOLDERS['build']/'runtime')]:
        target=BASE/'runtime'/role;shutil.copytree(source,target)
        assert not list(target.glob('LstmModelReplay.*'))
        for name,wanted in stage['products'][role].items():assert pin(target/name)==wanted,name
        for suffix in ['dll','deps.json','runtimeconfig.json']:
            name='LstmModelReplay.'+suffix;original=FOLDERS['replay']/'runtime/selected'/name
            shutil.copy2(original,target/name);assert pin(target/name)==pin(original)
        assert pin(target/'LstmModelReplay.dll')==stage['consumer']
    environment=read(FOLDERS['build']/'payload.json')
    for name,wanted in environment['external'].items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=read(FOLDERS['numerics']/'collection.json')['identities'][0],boot_time=1789634288.0,
        products=stage['products'],cores={role:files['Lokad.Onnx.dll'] for role,files in stage['products'].items()},consumer=stage['consumer'],
        external=environment['external'],interpreter=environment['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Exact current/candidate replay in four execution modes with unchanged qualified diagnostic consumer; all original tensors/native/ownership/scratch checks, no timing or rebuild.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))

if __name__=='__main__':main()
