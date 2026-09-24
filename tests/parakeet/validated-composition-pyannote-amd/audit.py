"""Reconcile every source, tensor, public request, resource and terminal worker."""
import copy
import json
from pathlib import Path
from run import BASE,ROOT,prepared
from protocol import JOBS,LIMITS,check_sample,pin,read,save
from prepare import APP_PAYLOAD,MODEL,OLD_DATA,NEW_DATA
from checks import inventory,model


def provenance(payload,collected):
    original=read(MODEL/'evidence/original-manifest.json')
    assert read(collected/'evidence/original-manifest.json')==original
    for role in ['selected','candidate']:
        expected=copy.deepcopy(original)
        expected.update(core_sha256=payload['identities'][role]['Lokad.Onnx.dll']['sha256'],data_sha256=payload['identities'][role]['Lokad.Onnx.Data.dll']['sha256'])
        expected['product_source']='M66 selected Core672e5f30/Data065b7a7f' if role=='selected' else 'M66 composition Core37c24375/Data cc37b19e'
        assert read(collected/'manifests'/(role+'-pyannote.json'))==expected
    assert (collected/'evidence/original-consumer.cs').read_bytes()==(MODEL/'consumer/Program.cs').read_bytes()
    assert (BASE/'bundle/consumer/Program.cs').read_bytes()==(MODEL/'consumer/Program.cs').read_bytes().replace(OLD_DATA.encode(),NEW_DATA.encode())
    original_payload=read(APP_PAYLOAD/'payload.json')
    for name,wanted in payload['files'].items():
        if name.startswith(('assets/','graph-reference/')) or name=='graph-reference.json':
            assert original_payload['files'][name]==wanted,name
    assert read(collected/'graph-reference.json')==read(APP_PAYLOAD/'graph-reference.json')
    for row in read(collected/'graph-reference.json'):assert pin(collected/row['path'])=={k:row[k] for k in ['bytes','sha256']}


def main():
    spec=prepared();assert not (BASE/'closed.json').exists()
    collected=BASE/'collected';receipt=read(collected/'collection.json');transfer=read(BASE/'collection-transfer.json');payload=read(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    provenance(payload,collected)
    state=read(collected/'identity.json');assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS and state['boot_time']==1789634288.0
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=[]
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<900
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']-state['started']<4*3600
    built=read(collected/'built.json');assert built['passed']
    for name,wanted in built['files'].items():
        if name.startswith('built/'):assert pin(collected/name)==wanted,name
    assert built['consumer']==pin(collected/'built/GraphQualification.dll')
    for name,wanted in payload['identities']['candidate'].items():assert built['files']['consumer/bin/Release/net10.0/'+name]==wanted
    il=inventory(read(collected/'consumer-inventory/instructions.json'),payload,built)
    assert il==read(collected/'consumer-inventory/review.json')
    results={}
    for role in ['selected','candidate']:
        results[role]=model(collected,role);assert results[role]==read(collected/role/'review.json')
    analysis=dict(passed=True,identities=payload['identities'],consumers=dict(payload['consumers'],candidate=built['consumer']),
        inventory=il,results=results,resources=resources,reference_provenance_verified=True,no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,files=files,local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__=='__main__':main()
