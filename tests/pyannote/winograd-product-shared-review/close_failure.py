"""Independently reconcile every shared/e5 tensor, native reference and VM worker."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'winograd-product-shared-amd'))
import copy
import json
from run import BASE, prepared
from prepare import ROOT, OLD
from protocol import JOBS, LIMITS, check_sample, pin, read, save
from checks import qualify


def provenance(payload, collected):
    original=read(OLD/'closed.json')['files']
    for name,entry in read(collected/'provenance.json').items():
        if name.startswith(('reference/','e5/')) or name in ['evidence/shared-historical.json','evidence/e5-historical.json']:
            wanted={k:entry[k] for k in ['bytes','sha256']}
            assert pin(collected/name)==wanted==original[entry['source']]
    for name,wanted in payload['model_assets'].items():
        relative=name.removeprefix('/home/vermorel/Onnx/')
        assert original[relative]==wanted==pin(ROOT/relative)


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    collected = BASE/'collected'; receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json'); payload = read(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 1 and receipt['input_error'] is None and receipt['payload'] == pin(BASE/'payload.json')
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    provenance(payload, collected)
    state = read(collected/'identity.json')
    assert state['complete'] and state['code'] == 1 and state['supervisor'] == read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']] == JOBS[:3] and payload['jobs'] == JOBS and state['boot_time'] == 1789634288.0
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert row['preflight'] == row['preflight_observations'][-1]
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        samples = [json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert len(sample['members']) <= 1
            assert all(row['members'][str(m['pid'])] == m['birth'] for m in sample['members'])
        gaps = [samples[0]['seconds']]+[b['seconds']-a['seconds'] for a, b in zip(samples, samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0 <= gap < 10 for gap in gaps)
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    assert state['ended']-state['started'] < 4*3600
    results = {name:qualify(collected,name,payload) for name in JOBS[:2]}
    for name,value in results.items():assert value == read(collected/name/'review.json')
    try: qualify(collected,'candidate-shared',payload)
    except AssertionError: pass
    else: raise AssertionError('The unchanged original exactness check must fail')
    assert 'checks.py' in state['error'] and 'path.read_bytes() ==' in state['error']
    candidate=read(collected/'candidate-shared/output/result.json')
    selected=read(collected/'selected-shared/output/result.json')
    assert candidate['passed'] and candidate['inputs_unchanged'] and candidate['held_outputs_unchanged']
    changed=[]
    for a,b in zip(candidate['rows'],selected['rows'],strict=True):
        assert all(a[k]==b[k] for k in ['model','scenario','step','name','shape','values','reference_file','reference_sha256'])
        if a['sha256'] != b['sha256']:changed.append({k:a[k] for k in ['model','scenario','step','name','shape','values','max_scaled_error','failed_values']})
    assert len(changed)==2 and all(r['model']=='resnet50' for r in changed)
    assert not (collected/'candidate-e5').exists()
    analysis=dict(passed=False,retained_failure=True,identities=payload['identities'],consumer=payload['consumer'],
        completed_jobs=JOBS[:3],missing_jobs=JOBS[3:],selected_checks=results,changed_rows=changed,
        resources=resources,reference_provenance_verified=True,no_performance_measurement=True,
        reason='Generic Winograd arithmetic affects eligible ResNet-50 convolutions. The original exact-all-shared contract fails; retain this failure. Candidate e5 never ran.')
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=False,retained_failure=True,files=files,local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json'),auditor=pin(Path(__file__))))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),passed=False,retained_failure=True,changed_rows=changed,resources=resources)))


if __name__=='__main__':main()
