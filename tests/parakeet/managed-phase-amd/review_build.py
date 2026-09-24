"""Admit the observer only after exact original-method and resource review."""
import base64
import json
from run import BASE, PRIOR, PRELUDE, pin, read, write, ssh


def main():
    folder=BASE/'build-collected';spec=read(BASE/'bundle/spec.json')
    receipt=read(folder/'build-collection.json');assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/'build-state.json');assert state['complete'] and state['code']==0
    resources=[]
    for run in state['runs']:
        assert run['complete'] and run['code']==0
        limits=spec['build_limits']
        assert run['preflight']['available']>=limits['available_before'] and run['preflight']['tmpfs']>=limits['tmpfs_before']
        rows=[json.loads(line) for line in (folder/'logs'/(run['name']+'.resources.jsonl')).read_text().splitlines()]
        assert len(rows)==run['samples'] and rows
        for row in rows:
            assert row['seconds']<limits['seconds'] and row['rss']<limits['rss']
            assert row['available']>=spec['minimum_free'] and row['tmpfs']>=spec['minimum_free'] and row['output']<spec['output_limit']
            assert row['rss']==sum(m['rss'] for m in row['members'])
            for m in row['members']:
                assert run['members'][str(m['pid'])]==m['birth'] and m['affinity']==[2] and all(t==[2] for t in m['threads'])
        resources.append(dict(name=run['name'],samples=len(rows),peak_rss=max(r['rss'] for r in rows)))
    built=read(folder/'built.json')
    assert built['core']==spec['core']
    for name,wanted in built['runtime_files'].items():assert pin(folder/name)==wanted,name
    inventory=read(folder/'inventory/instructions.json');assert inventory['inventory_complete']
    reviews=[]
    for row in inventory['observations']:
        name=row['assembly'];assert name in ['SampledAudio.dll','Lokad.Onnx.Data.dll']
        assert row['public_surface_equal'] and not row['removed']
        assert row['before_sha256']==pin(PRIOR/'collected/runtime'/name)['sha256']
        assert row['after_sha256']==pin(folder/'runtime-observed'/name)['sha256']
        for key,flags in row['method_flags_before'].items():assert row['method_flags_after'][key]==flags,key
        key,=row['differences']
        if name=='SampledAudio.dll':
            assert row['methods']==162 and row['unchanged_methods']==161
            assert key.startswith('Program::<Main>$::')
            assert len(row['added'])==2 and all(k.startswith('PhaseConsumer::') for k in row['added'])
        else:
            assert row['methods']==697 and row['unchanged_methods']==696
            assert key.startswith('Lokad.Onnx.ParakeetTranscriber::Execute::')
            assert all(k.startswith(('Lokad.Onnx.ParakeetPhaseProbe','<>f__AnonymousType')) for k in row['added'])
            before=json.loads(row['normalized_methods'][key]);after=json.loads(row['candidate_methods'][key])
            assert before['InitLocals']==after['InitLocals'] and before['MaxStackSize']==after['MaxStackSize']
            body=[dict(i,offset=i['offset']-7) for i in after['instructions'] if 7<=i['offset']<120]
            assert body==before['instructions'][:-1], 'Original Execute arithmetic/branches changed'
            assert [i['opcode'] for i in after['instructions'][:3]]==['ldarg.0','call','stloc.0']
            assert after['instructions'][1]['operand']=='Lokad.Onnx.ParakeetPhaseProbe::Scope Enter(Lokad.Onnx.GraphExecution)'
            assert [i['opcode'] for i in after['instructions'] if i['offset']>=120]==['stloc.1','leave.s','ldloca.s','constrained.','callvirt','endfinally','ldloc.1','ret']
            assert after['exceptions']==[dict(flags=2,TryOffset=7,TryLength=116,HandlerOffset=123,HandlerLength=14,filter=-1,caught=None)]
        reviews.append(dict(assembly=name,original_methods=row['methods'],unchanged=row['unchanged_methods'],changed=key,
            added=len(row['added']),public_surface_equal=True,original_flags_equal=True))
    value=dict(passed=True,built=pin(folder/'built.json'),inventory=pin(folder/'inventory/instructions.json'),
        collection=pin(folder/'build-collection.json'),spec=pin(BASE/'bundle/spec.json'),
        core_unchanged=True,methods=reviews,resources=resources,reviewer=pin(__file__))
    write(BASE/'build-review.json',value)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    remote=ssh(PRELUDE+f'''
import base64
from remote import verify,pin,read,live
verify();state=read(base/'build-state.json');assert state['complete'] and state['code']==0
assert not live(state['supervisor']) and all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={value['built']!r} and pin(base/'inventory/instructions.json')=={value['inventory']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert remote['review']==pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json',remote)
    print(json.dumps(dict(passed=True,review=remote['review'],methods=reviews,core=built['core'],data=built['data'],consumer=built['consumer'])))


if __name__=='__main__':main()
