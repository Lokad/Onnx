"""Pin exact completed candidate binaries and the verified offline toolchain."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
BUILD=Path('/dev/shm/lokad-pyannote-lstm-wide-build-20260922')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    closed=read(BASE/'evidence/build-closed.json')
    assert pin(BUILD/'collection.json')==closed['files']['collected/collection.json']
    receipt=read(BUILD/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for identity in receipt['identities']:assert not live(identity)
    for name,wanted in receipt['files'].items():assert pin(BUILD/name)==wanted,name
    assert pin(BUILD/'payload.json')['sha256']=='df6f63dd25481ca7d9781811407b998528e9591ad631a2d8f476e5ff9151eed5'
    previous=read(BUILD/'payload.json');shutil.copytree(BUILD/'runtime',BASE/'runtime')
    for name,wanted in stage['product'].items():assert pin(BASE/'runtime'/name)==wanted,name
    for name,wanted in previous['external'].items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        product=stage['product'],feed=previous['feed'],external=previous['external'],interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        expected_exit={name:1 if name=='suite-scalar' else 0 for name in JOBS},
        scope='Focused full LSTM census against exact built product; original eighteen scalar explicit-intrinsics refusals retained; no timing.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
