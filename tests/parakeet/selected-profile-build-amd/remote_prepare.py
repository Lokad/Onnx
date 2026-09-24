"""Verify terminal selected-product prerequisites and reuse the exact offline environment."""
import json
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
PREREQUISITES = {
    'app': '/dev/shm/lokad-parakeet-wide-entry-first-use-app-v2-20260923',
    'control': '/dev/shm/lokad-parakeet-dense-scalar-where-balanced-control-20260924'}


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name,wanted in stage['files'].items(): assert pin(BASE/name) == wanted,name
    receipts = []
    for label,path in PREREQUISITES.items():
        folder=Path(path)
        assert pin(folder/'collection.json') == pin(BASE/'evidence'/(label+'-collection.json'))
        receipt=read(folder/'collection.json'); receipts.append(receipt)
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert all(not live(identity) for identity in receipt['identities'])
        assert pin(folder/'payload.json') == pin(BASE/'evidence'/(label+'-payload.json'))
        for name,wanted in read(folder/'payload.json')['files'].items(): assert pin(folder/name)==wanted,name
    for name,wanted in stage['product'].items(): assert pin(BASE/'runtime'/name)==wanted,name
    assert pin(BASE/'previous/SampledAudio.dll') == stage['previous_consumer']
    environment = read(BASE/'evidence/control-payload.json')
    for name,wanted in environment['external'].items(): assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,product=stage['product'],previous_consumer=stage['previous_consumer'],
        previous_owner=receipts[-1]['identities'][0],boot_time=1789634288.0,feed=environment['feed'],
        external=environment['external'],interpreter=environment['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Selected release copied unchanged; only two selected-product hash literals change in SampledAudio. Complete IL and implementation-flag inventory required.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
