"""Admit only a corrected test consumer of the already inspected Core."""
import base64
import json
from run import BASE,RECOVERY,PRELUDE,pin,read,write,ssh,prepared
from checks import resources


def failed_capture():
    folder=RECOVERY/'capture-collected';state=read(folder/'capture-state.json')
    receipt=read(folder/'capture-collection.json');spec=read(RECOVERY/'bundle/spec.json');limits=spec['capture_limits']
    transfer=read(RECOVERY/'capture-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(RECOVERY/'capture-results.tar.gz') and transfer['collection']==pin(folder/'capture-collection.json')
    assert receipt['terminal'] and receipt['code']==state['code']==1 and state['complete']
    assert receipt['state']==pin(folder/'capture-state.json') and state['supervisor']==read(RECOVERY/'capture-deployment.json')
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    row,=state['runs'];assert row['name']=='tensors-512' and row['complete'] and row['code']==1
    assert row['seconds']<limits['seconds'] and row['preflight']['available']>=limits['available_before'] and row['preflight']['tmpfs']>=limits['tmpfs_before']
    samples=[json.loads(s) for s in (folder/'logs/tensors-512.resources.jsonl').read_text().splitlines()]
    assert len(samples)==row['samples']>0
    for s in samples:
        assert 0<=s['seconds']<limits['seconds'] and s['rss']==sum(m['rss'] for m in s['members'])<limits['rss']
        assert s['available']>=spec['minimum_free'] and s['tmpfs']>=spec['minimum_free'] and s['output']<spec['output_limit']
        for m in s['members']:assert row['members'][str(m['pid'])]==m['birth'] and m['affinity']==[2] and all(t==[2] for t in m['threads'])
    gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
    assert all(0<=g<10 for g in gaps)
    return dict(collection=pin(folder/'capture-collection.json'),passed=394,failed=1,samples=len(samples),peak_rss=max(s['rss'] for s in samples))


def main():
    prepared();assert not (BASE/'build-review.json').exists()
    resource_rows=resources('build');failure=failed_capture();folder=BASE/'build-collected';built=read(folder/'built.json');spec=read(BASE/'bundle/spec.json')
    assert [r['name'] for r in resource_rows]==['tensors-restore','tensors-build']
    for name,wanted in built['runtime_files'].items():assert pin(folder/name)==wanted,name
    assert built['core']==spec['core']==read(RECOVERY/'build-collected/built.json')['core'] and not built['product_rebuilt']
    assert built['consumer']!=read(RECOVERY/'build-collected/built.json')['consumer']
    assert spec['compiled_review']==pin(RECOVERY/'build-review.json') and read(RECOVERY/'build-review.json')['passed']
    for path in (folder/'logs').glob('*.stdout'):
        log=path.read_text(encoding='utf8');assert ': warning ' not in log and '  Lokad.Onnx ->' not in log
    value=dict(passed=True,built=pin(folder/'built.json'),compiled_review=spec['compiled_review'],core=built['core'],
        product_rebuilt=False,corrected_test=spec['corrected_test'],source_project_binding=spec['source_project_binding'],
        resources=resource_rows,original_failed_suite=failure,reviewer=pin(__file__))
    write(BASE/'build-review.json',value)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    result=ssh(PRELUDE+f'''
import base64
from remote import verify,read,pin,live,idle
verify();idle();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={value['built']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert result['review']==pin(BASE/'build-review.json');write(BASE/'build-review-transferred.json',result)
    print(json.dumps(dict(passed=True,review=result['review'],core=built['core'],consumer=built['consumer'],product_rebuilt=False)))


if __name__=='__main__':main()
