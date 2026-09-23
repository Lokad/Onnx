"""Audit all consumer method comparisons and eight complete AMD numerical results."""
import json
from run import BASE, prepared
from protocol import JOBS, LIMITS, check_sample, pin, read, save
from consumer_checks import consumer_inventory
from checks import check_result


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    c = BASE/'collected'; receipt = read(c/'collection.json')
    transfer = read(BASE/'collection-transfer.json'); payload = read(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(c/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None and receipt['payload'] == pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items(): assert pin(c/name) == wanted,name
    for name,wanted in read(BASE/'bundle/stage.json')['files'].items():
        assert payload['files'][name] == wanted and pin(BASE/'bundle'/name) == wanted,name
    state = read(c/'identity.json')
    assert state['complete'] and state['code'] == 0 and state['supervisor'] == read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS and state['boot_time'] == 1789634288.0
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    built = read(c/'consumer-built.json'); assert built['passed'] and set(built['consumers']) == {'raw','wide','span','channels64','channels128','layers'}
    for name,wanted in built['files'].items(): assert pin(c/name) == wanted,name
    actual = dict(payload,consumers=built['consumers']); resources=[]; reports={}; inventories={}
    for row in state['runs']:
        name=row['name']; mode,action=name.split('-')
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        assert row['preflight'] == row['preflight_observations'][-1] == read(c/(name+'-preflight.json'))[-1]
        samples = [json.loads(s) for s in (c/'logs'/(name+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])] == m['birth'] for m in sample['members'])
        resources.append(dict(name=name,samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
        if action == 'build':
            assert built['files']['consumers/'+mode+'/bin/Release/net10.0/Lokad.Onnx.dll'] == payload['core']
        if action == 'inventory':
            inventories[mode]=consumer_inventory(read(c/name/'instructions.json'),mode,payload['prior_consumers'][mode],built['consumers'][mode],payload['core']['sha256'])
            assert inventories[mode] == read(c/name/'review.json')
        if action in ['256','512']:
            result=read(c/name/'result.json'); assert result['pid'] == row['child']['pid'] and result['runtime'] == '10.0.8'
            reports[name]=check_result(result,mode,action,actual,BASE/'bundle')
            assert reports[name] == read(c/name/'review.json')
    assert state['ended']-state['started'] < 4*3600
    analysis=dict(passed=True,core=payload['core'],consumers=built['consumers'],inventories=inventories,reports=reports,
        resources=resources,samples=sum(r['samples'] for r in resources),peak_rss=max(r['peak_rss'] for r in resources),
        no_performance_measurement=True,codegen_pending=True,root_product_changed=False)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,files=files,local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__ == '__main__': main()
