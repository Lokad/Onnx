"""Prove the observer's compiled scope before any model capture."""
import base64
import json
from run import BASE, PRELUDE, CORE_SHA, pin, read, write, ssh, prepared
from checks import consumer_scope, data_scope


def resources(kind):
    folder = BASE/(kind+'-collected'); spec = read(BASE/'bundle/spec.json')
    receipt = read(folder/(kind+'-collection.json')); assert receipt['terminal'] and receipt['code'] == 0
    transfer = read(BASE/(kind+'-transfer.json'))
    assert transfer['passed'] and transfer['archive'] == pin(BASE/(kind+'-results.tar.gz'))
    assert transfer['collection'] == pin(folder/(kind+'-collection.json'))
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    state = read(folder/(kind+'-state.json')); assert state['complete'] and state['code'] == 0
    assert state['supervisor'] == read(BASE/(kind+'-deployment.json'))
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
    prepared()
    resource_rows = resources('build'); folder = BASE/'build-collected'
    assert [r['name'] for r in resource_rows] == ['sdk-version','data-restore','data-build','consumer-restore',
        'consumer-build','bridge-restore','bridge-build','inventory']
    spec = read(BASE/'bundle/spec.json'); built = read(folder/'built.json')
    assert built['core'] == spec['core']
    for name, wanted in built['runtime_files'].items(): assert pin(folder/name) == wanted, name
    inventory = read(folder/'inventory/instructions.json'); assert inventory['inventory_complete']
    assert {r['assembly'] for r in inventory['observations']} == {'Lokad.Onnx.Data.dll','SampledAudio.dll'}
    assert len(inventory['observations']) == 2
    reviews = []
    for row in inventory['observations']:
        name = row['assembly']
        assert row['public_surface_equal'] and not row['removed']
        assert row['before_sha256'] == pin(BASE/'bundle/reference-runtime'/name)['sha256']
        assert row['after_sha256'] == pin(folder/'runtime-observed'/name)['sha256']
        for key, flags in row['method_flags_before'].items(): assert row['method_flags_after'][key] == flags, key
        key, = row['differences']
        before = json.loads(row['normalized_methods'][key]); after = json.loads(row['candidate_methods'][key])
        if name == 'SampledAudio.dll':
            assert row['methods'] == 162 and row['unchanged_methods'] == 161
            assert key.startswith('Program::<Main>$::')
            assert len(row['added']) == 2 and all(k.startswith('MaskingConsumer::') for k in row['added'])
            consumer_scope(before, after, CORE_SHA)
        else:
            assert row['methods'] == 697 and row['unchanged_methods'] == 696
            assert key.startswith('Lokad.Onnx.ParakeetTranscriber::Execute::')
            assert row['added'] and all(k.startswith('Lokad.Onnx.ParakeetMaskingProbe') for k in row['added'])
            data_scope(before, after)
        reviews.append(dict(assembly=name,original_methods=row['methods'],unchanged=row['unchanged_methods'],changed=key,
            added=len(row['added']),original_body_preserved=True,public_surface_equal=True,original_flags_equal=True))
    value = dict(passed=True,built=pin(folder/'built.json'),inventory=pin(folder/'inventory/instructions.json'),
        collection=pin(folder/'build-collection.json'),spec=pin(BASE/'bundle/spec.json'),core_unchanged=True,
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
    print(json.dumps(dict(passed=True,review=remote['review'],methods=reviews,core=built['core'],data=built['data'])))


if __name__ == '__main__': main()
