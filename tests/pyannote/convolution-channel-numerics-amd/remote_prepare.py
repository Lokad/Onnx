"""Verify the completed product, original qualified probes and all native fixtures."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
PRODUCT=Path('/dev/shm/lokad-pyannote-convolution-channel-build-20260923')
PRIOR=Path('/dev/shm/lokad-pyannote-spatial-weight-reuse-20260922')

def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    product_proof=read(BASE/'evidence/product-closed.json');old_proof=read(BASE/'evidence/previous-amd-closed.json')
    assert pin(PRODUCT/'payload.json')==product_proof['files']['payload.json']
    assert pin(PRIOR/'payload.json')==old_proof['files']['payload/payload.json']
    for folder,proof in [(PRODUCT,product_proof),(PRIOR,old_proof)]:
        assert pin(folder/'collection.json')==proof['files']['collected/collection.json']
        receipt=read(folder/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        for identity in receipt['identities']:assert not live(identity)
        for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    product=read(PRODUCT/'payload.json');prior=read(PRIOR/'payload.json')
    shutil.copytree(PRODUCT/'runtime',BASE/'runtime')
    for name,wanted in stage['product'].items():assert pin(BASE/'runtime'/name)==wanted,name
    external=dict(product['external'])
    for name,wanted in prior['external'].items():
        assert external.get(name,wanted)==wanted;external[name]=wanted
    for name,wanted in prior['files'].items():
        if name.startswith('runtime/'):external[str(PRIOR/name)]=wanted
    prior_runtimes={mode:str(PRIOR/'runtime'/('raw' if mode.startswith('channels') else mode)) for mode in stage['prior_consumers']}
    for mode,wanted in stage['prior_consumers'].items():
        assembly='LayerGraphs.dll' if mode=='layers' else 'Lokad.Onnx.Backend.Tests.dll'
        assert pin(Path(prior_runtimes[mode])/assembly)==wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=read(PRODUCT/'collection.json')['identities'][0],boot_time=1789634288.0,
        core=stage['core'],product=stage['product'],prior_consumers=stage['prior_consumers'],prior_runtimes=prior_runtimes,
        fixture_directory=prior['fixture_directory'],feed=product['feed'],external=external,interpreter=product['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Six consumers: four exact inherited families and two new channel lists; twelve AMD numerical workers, no timing.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))

if __name__=='__main__':main()
