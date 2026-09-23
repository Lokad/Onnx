"""Copy unchanged products and qualified GraphRaw consumer after terminal proof."""
import json,shutil
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
BASE=Path(__file__).resolve().parents[1]
FOLDERS=dict(build=Path('/dev/shm/lokad-pyannote-convolution-channel-build-20260923'),
    numerical=Path('/dev/shm/lokad-pyannote-convolution-channel-numerics-20260923'),current=Path('/dev/shm/lokad-parakeet-current-baseline-20260922'))

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
    for role,source in [('selected',FOLDERS['current']/'runtimes/current'),('candidate',FOLDERS['build']/'runtime')]:
        target=BASE/'runtime'/role;shutil.copytree(source,target)
        for name,wanted in stage['products'][role].items():assert pin(target/name)==wanted,name
        probe=FOLDERS['numerical']/'consumers/raw/bin/Release/net10.0/Lokad.Onnx.Backend.Tests.dll'
        assert pin(probe)==stage['probe'];shutil.copy2(probe,target/probe.name)
    environment=read(FOLDERS['build']/'payload.json')
    for name,wanted in environment['external'].items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,products=stage['products'],probe=stage['probe'],
        previous_owner=read(FOLDERS['numerical']/'collection.json')['identities'][0],boot_time=1789634288.0,
        feed=environment['feed'],external=environment['external'],interpreter=environment['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Build new finite-extreme driver only; exact qualified graph caller and both products; 1280 cases per role/width, no timing.')
    save(BASE/'payload.json',payload);verify(BASE);print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))

if __name__=='__main__':main()
