"""Independent checks for terminal conversion owners and every resource sample."""
from common import *


def check_sample(sample,identity):
    assert 0<=sample['seconds']<900 and 0<=sample['rss']<8*1024**3
    assert sample['available']>=1024**3 and sample['disk']>=1024**3
    assert 0<=sample['output_bytes']<=1024**3 and 0<=sample['artifacts']<=2*1024**3
    assert sample['rss']==sum(m['rss'] for m in sample['members'])
    assert len(sample['members'])<=1
    for member in sample['members']:
        assert {k:member[k] for k in ['pid','birth']}==identity
        assert member['affinity']==[0] and member['threads']
        assert all(t['affinity']==[0] for t in member['threads'])


def inspect():
    target=BASE/'exports';receipt=read(target/'collection.json');transfer=read(BASE/'export-transfer.json')
    assert transfer['terminal'] and transfer['code']==0 and transfer['input_error'] is None
    assert transfer['archive']==pin(BASE/'export-results.tar.gz') and transfer['receipt']==pin(target/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload/payload.json') and receipt['capture_receipt']==pin(BASE/'collected/collection.json')
    for name,wanted in receipt['files'].items():assert pin(target/name)==wanted,name
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}==set(receipt['files'])|{'collection.json'}
    state=read(target/'identity.json');deployment=read(BASE/'export-deployment.json')
    assert deployment==read(target/'export-deployment.json')==state['supervisor']
    assert state['complete'] and state['code']==0 and state['boot_time']==1789634288.0
    jobs=[(n,f) for n in ['sampled-a','sampled-b'] for f in ['Speedscope','Chromium']]
    assert [r['name'] for r in state['runs']]==[n+'-'+f.lower() for n,f in jobs]
    assert receipt['identities']==[deployment]+[r['identity'] for r in state['runs']]
    resources=[]
    for row,(capture,format) in zip(state['runs'],jobs):
        assert row['complete'] and row['code']==0 and row['seconds']<900
        assert row['preflight']['available']>=8*1024**3 and row['preflight']['disk']>=3*1024**3
        assert row['command']==[DOTNET,REMOTE+'/tracer/dotnet-trace.dll','convert',REMOTE+'/'+capture+'/capture.nettrace','--format',format,'--output',REMOTE+'/exports/'+capture+'/'+format.lower()]
        samples=[json.loads(s) for s in (target/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        assert samples[-1]['seconds']<=row['seconds']
        for sample in samples:check_sample(sample,row['identity'])
        expected=capture+'/'+format.lower()+'.'+format.lower()+'.json'
        assert row['output']==dict(path=expected,**pin(target/expected))
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']>=state['started'] and state['ended']-state['started']<4*900+120
    return dict(resources=resources,identities=receipt['identities'])
