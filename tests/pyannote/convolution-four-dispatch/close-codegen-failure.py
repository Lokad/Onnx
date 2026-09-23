"""Preserve the diagnostic-filter failure before inference, with complete ownership proof."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/pyannote/convolution-four-codegen-amd'))
from run import BASE,prepared
from protocol import pin,read,save,check_sample

spec=prepared();assert not (BASE/'failure.json').exists()
c=BASE/'collected';receipt=read(c/'collection.json');transfer=read(BASE/'collection-transfer.json')
assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None
assert receipt['payload']==pin(BASE/'payload.json')
for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
state=read(c/'identity.json');assert state['complete'] and state['code']==1
assert state['supervisor']==read(BASE/'deployment.json')
row,=state['runs'];assert row['name']=='production' and row['complete'] and row['code']==-6
assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for p,b in row['members'].items()]
samples=[json.loads(s) for s in (c/'logs/production.jsonl').read_text().splitlines()]
assert len(samples)==row['samples'] and max(s['rss'] for s in samples)==row['peak_rss']
for sample in samples:check_sample(sample)
error=(c/'logs/production.stderr').read_text()
assert 'System.IO.InvalidDataException: Diagnostic flag allowlist' in error
assert not (c/'production/result.json').exists() and not (c/'candidate').exists()
driver=ROOT/'tests/pyannote/spatial-weight-codegen/Driver.cs';source=driver.read_text()
assert source.index('"Diagnostic flag allowlist"')<source.index('var assembly = Assembly.LoadFrom')
files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
save(BASE/'failure.json',dict(passed=False,files=files,terminal=receipt['identities'],
    before_inference=True,candidate_launched=False,reason='Broader method filter refused by exact unchanged driver allowlist.',
    driver_source=pin(driver),resource_samples=len(samples),peak_rss=row['peak_rss'],
    recovery='Separate v2 capture restores original Kernel512* filter; source and complete compiled inventory prove Execute dispatch. No Execute machine-code claim.'))
print(json.dumps(dict(failure=pin(BASE/'failure.json'),before_inference=True,candidate_launched=False)))
