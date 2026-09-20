"""Read-only coverage, identity, numerical-assertion, resource and timing audit."""
import argparse
from common import *


def audit(base):
    payload=base/'payload';collected=base/'collected';bundle=read(payload/'bundle.json');sched=read(payload/'schedule.json')
    assert sched==schedule() and bundle['limits']==LIMITS and bundle['core']==CORE and bundle['probe']==PROBE and bundle['proof_receipt']==PROOF
    for name,wanted in bundle['files'].items():assert pin(payload/name)==wanted,name
    receipt=read(collected/'collection.json');assert receipt['complete'] and receipt['code']==0 and receipt['checkout']=='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887'
    assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()}==set(receipt['files'])|{'collection.json'}
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    assert pin(collected/'bundle.json')==pin(payload/'bundle.json')
    for name,wanted in bundle['files'].items():
        if not name.startswith('bin/'):assert pin(collected/name)==wanted,name
    deployment=read(base/'deployment.json');identity=read(collected/'result/identity.json')
    assert deployment==read(collected/'deployment.json') and deployment['bundle_sha256']==pin(payload/'bundle.json')['sha256']==identity['bundle_sha256']
    assert (deployment['pid'],deployment['start'])==(identity['supervisor']['pid'],identity['supervisor']['start'])
    samples={r['name']:[json.loads(line) for line in (collected/'result'/(r['name']+'-samples.jsonl')).read_text().splitlines()] for r in identity['runs']}
    resources=validate_resources(identity,samples);workers=[]
    for worker in range(4):
        directory=collected/'result'/f'worker{worker}';i=read(directory/'identity.json');run=identity['runs'][worker]
        assert i['worker']==worker and i['pid']==run['pid'] and i['runtime']=='10.0.8' and i['affinity']==4 and i['processor_count']==1 and i['avx512'] and i['flags']=={}
        assert i['core']==CORE and i['probe']==PROBE and i['consumer']==bundle['files']['bin/Timing.dll']['sha256']
        assert i['schedule']==bundle['files']['schedule.json']['sha256'] and i['packed_bytes']==84934656 and i['frequency']>0 and i['weight_setup_seconds']>0
        assert [(w['n'],w['k']) for w in i['weights']]==[(384,384),(384,384),(384,384),(384,384),(384,1536),(1536,384)]*12
        assert read(directory/'complete.json')==dict(complete=True,worker=worker,pid=i['pid'])
        assert {p.name for p in directory.iterdir()}=={b['name']+'.json' for b in BANKS}|{'identity.json','complete.json'}
        raw=[read(directory/(b['name']+'.json')) for b in BANKS];banks=[]
        for index,bank in enumerate(raw):
            banks.append(validate_bank(bank,worker,index,i['frequency'],sched))
            assert bank['before']['packed']==[w['hash'] for w in i['weights']]
        chronological=[raw[index] for index in ORDERS[worker]]
        assert all(a['measured'][-1]['end']<b['first'][0]['start'] for a,b in zip(chronological,chronological[1:]))
        stdout=[json.loads(line) for line in (collected/'result'/f'worker{worker}.stdout').read_text().splitlines()]
        assert stdout==[dict(worker=worker,bank=bank['name'],conditioning_cycles=bank['conditioning_cycles'],measured=192) for bank in chronological]
        assert not (collected/'result'/f'worker{worker}.stderr').read_text().strip()
        workers.append(dict(identity=i,banks=banks,raw=raw))
    for index in range(5):assert all(w['raw'][index]['before']==workers[0]['raw'][index]['before'] for w in workers)
    performance=verdict(workers)
    return dict(source=bundle['source'],integrity_passed=True,resources=resources,performance=performance,
                workers=[dict(identity=w['identity'],banks=w['banks']) for w in workers],
                counts=dict(first=80,measured=3840,conditioning=sum(b['conditioning_calls'] for w in workers for b in w['banks'])),
                measured_total_allocated=sum(b['total_allocated'] for w in workers for b in w['banks']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();result=audit(a.artifact.resolve());write(a.artifact/'audit.json',result)
    print(json.dumps(dict(counts=result['counts'],verdict=result['performance'],resources=result['resources'])))
