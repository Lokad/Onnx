"""Require a single scoped Core observation call before layout capture."""
import base64
import json
from run import BASE, PRIOR, PRELUDE, pin, read, write, ssh


def resources(kind):
    folder = BASE/(kind+'-collected'); spec = read(BASE/'bundle/spec.json')
    receipt = read(folder/(kind+'-collection.json')); assert receipt['terminal'] and receipt['code'] == 0
    for name,wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    state = read(folder/(kind+'-state.json')); assert state['complete'] and state['code'] == 0
    limits = spec[kind+'_limits']; result = []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < limits['seconds']
        assert run['preflight']['available'] >= limits['available_before'] and run['preflight']['tmpfs'] >= limits['tmpfs_before']
        rows = [json.loads(line) for line in (folder/'logs'/(run['name']+'.resources.jsonl')).read_text().splitlines()]
        assert rows and len(rows) == run['samples']
        for row in rows:
            assert row['seconds'] < limits['seconds'] and row['rss'] < limits['rss']
            assert row['available'] >= spec['minimum_free'] and row['tmpfs'] >= spec['minimum_free'] and row['output'] < spec['output_limit']
            assert row['rss'] == sum(m['rss'] for m in row['members'])
            for member in row['members']:
                assert run['members'][str(member['pid'])] == member['birth']
                assert member['affinity'] == [2] and all(t == [2] for t in member['threads'])
        assert all(0 <= b['seconds']-a['seconds'] < 10 for a,b in zip(rows,rows[1:]))
        result.append(dict(name=run['name'],samples=len(rows),peak_rss=max(r['rss'] for r in rows)))
    return result


def main():
    resource_rows = resources('build'); folder = BASE/'build-collected'
    spec = read(BASE/'bundle/spec.json'); built = read(folder/'built.json')
    assert built['data'] == spec['data']
    for name,wanted in built['runtime_files'].items(): assert pin(folder/name) == wanted, name
    inventory = read(folder/'inventory/instructions.json'); assert inventory['inventory_complete']
    reviews = []
    for row in inventory['observations']:
        name = row['assembly']; assert name in ['Lokad.Onnx.dll','SampledAudio.dll']
        assert row['public_surface_equal'] and not row['removed']
        assert row['before_sha256'] == pin(PRIOR/'collected/runtime'/name)['sha256']
        assert row['after_sha256'] == pin(folder/'runtime-observed'/name)['sha256']
        for key,flags in row['method_flags_before'].items(): assert row['method_flags_after'][key] == flags, key
        key, = row['differences']
        assert row['unchanged_methods'] == row['methods']-1
        if name == 'SampledAudio.dll':
            assert key.startswith('Program::<Main>$::')
            assert len(row['added']) == 2 and all(k.startswith('LayoutConsumer::') for k in row['added'])
        else:
            assert key.startswith('Lokad.Onnx.TensorSlice`1[T]::Reshape::'), key
            assert all(k.startswith('Lokad.Onnx.SliceLayoutProbe') for k in row['added'])
            before = json.loads(row['normalized_methods'][key]); after = json.loads(row['candidate_methods'][key])
            assert {k:v for k,v in before.items() if k != 'instructions'} == {k:v for k,v in after.items() if k != 'instructions'}
            assert [i['opcode'] for i in after['instructions'][:3]] == ['ldarg.0','ldarg.1','call']
            assert 'Lokad.Onnx.SliceLayoutProbe::Void Observe[T]' in after['instructions'][2]['operand']
            original = [dict(i,offset=i['offset']-7) for i in after['instructions'][3:]]
            assert original == before['instructions'], 'Original Reshape body changed'
        reviews.append(dict(assembly=name,original_methods=row['methods'],unchanged=row['unchanged_methods'],changed=key,
            added=len(row['added']),public_surface_equal=True,original_flags_equal=True))
    value = dict(passed=True,built=pin(folder/'built.json'),inventory=pin(folder/'inventory/instructions.json'),
        collection=pin(folder/'build-collection.json'),spec=pin(BASE/'bundle/spec.json'),data_unchanged=True,
        methods=reviews,resources=resource_rows,reviewer=pin(__file__))
    write(BASE/'build-review.json',value)
    encoded = base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    remote = ssh(PRELUDE+f'''
import base64
from remote import verify,pin,read,live
verify();state=read(base/'build-state.json');assert state['complete'] and state['code']==0
assert not live(state['supervisor']) and all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={value['built']!r} and pin(base/'inventory/instructions.json')=={value['inventory']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert remote['review'] == pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json',remote)
    print(json.dumps(dict(passed=True,review=remote['review'],methods=reviews,core=built['core'],data=built['data'],consumer=built['consumer'])))


if __name__ == '__main__': main()
