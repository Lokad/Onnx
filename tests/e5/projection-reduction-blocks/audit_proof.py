"""Check complete saved outputs, proof coverage, resources and actual code."""
import argparse, struct
from proof_common import *
from code_audit import inspect


def audit(base):
    bundle=read(base/'payload/bundle.json');collection=read(base/'collected/collection.json');collected=base/'collected'
    assert collection['complete'] and collection['code']==0 and collection['checkout']=='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887'
    for name,wanted in collection['files'].items():assert pin(collected/name)==wanted,name
    for name,wanted in bundle['files'].items():
        assert pin(base/'payload'/name)==wanted,name
        if not name.startswith('bin/'):assert pin(collected/name)==wanted,name
    assert pin(collected/'bundle.json')==pin(base/'payload/bundle.json')
    assert bundle['cases']==[list(row) for row in cases()] and bundle['limits']==LIMITS and bundle['core']==CORE and bundle['probe']==PROBE
    state=read(collected/'result/identity.json');assert state['bundle_sha256']==pin(base/'payload/bundle.json')['sha256']
    samples={name:[json.loads(line) for line in (collected/'result'/(name+'-samples.jsonl')).read_text().splitlines()] for name in ['plain','code']}
    resources=validate_resources(state,samples);rows=[];previous=None;expected_scalar=sum(m*10 for m,_,_,_ in cases())
    for run in state['runs']:
        name=run['name'];folder=collected/'result'/name;result=read(folder/'result.json')
        assert result['complete'] and result['supported'] and result['pid']==run['pid'] and result['affinity']==4 and result['processor_count']==1
        assert result['runtime']=='10.0.8' and result['core']==CORE and result['probe']==PROBE and result['refusals']==31 and result['scalar_values']==expected_scalar
        assert [(r['m'],r['n'],r['k'],r['exceptional']) for r in result['checks']]==cases()
        assert not (collected/'result'/(name+'.stderr')).read_text().strip()
        expected=dict(passed=True,cases=473,scalar_values=expected_scalar,refusals=31)
        assert read(collected/'result'/(name+'.stdout'))==expected
        assert set(p.name for p in folder.iterdir())=={'result.json'}|{f'{i:04}.f32' for i in range(473)}
        for i,row in enumerate(result['checks']):
            assert row['file']==f'{i:04}.f32' and row['values']==row['m']*row['k']
            path=folder/row['file'];assert pin(path)==dict(bytes=(row['values']+10)*4,sha256=row['output'])
            data=path.read_bytes();assert data[:20]==data[-20:]==struct.pack('<f',-12345.5)*5
            if not row['exceptional']:
                assert all((bits[0]&0x7f800000)!=0x7f800000 for bits in struct.iter_unpack('<I',data))
            assert all(len(row[key])==64 for key in ['input','weights','packed','output','twice'])
        if previous is not None:assert result['checks']==previous,'Code capture changed outputs'
        previous=result['checks'];rows.append(dict(name=name,cases=473,values=sum(r['values'] for r in result['checks']),scalar_values=expected_scalar))
    assert state['runs'][0]['flags']=={}
    assert state['runs'][1]['flags']['COMPlus_JitDisasm']=='*PackedTile12* *PackedTile8*'
    code=inspect((collected/'result/jit.txt').read_text())
    return dict(passed=True,bundle=pin(base/'payload/bundle.json'),resources=resources,proofs=rows,code=code)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();value=audit(a.artifact.resolve());write(a.artifact/'audit.json',value)
    print(json.dumps(dict(passed=True,proofs=value['proofs'],methods=[dict(method=r['method'],bytes=r['code_bytes']) for r in value['code']['methods']])))
