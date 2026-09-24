"""Reconcile the completed build, failed inspection and its bounded recovery."""
import json
from prepare import BUILD
from run import BASE, prepared
from protocol import LIMITS, check_sample, pin, read, save
from checks import inventory


def resources(base, jobs, codes):
    folder = base/'collected'; receipt = read(folder/'collection.json'); state = read(folder/'identity.json')
    transfer = read(base/'collection-transfer.json'); payload = read(base/'payload.json')
    assert transfer['passed'] and transfer['archive'] == pin(base/'results.tar.gz') and transfer['receipt'] == pin(folder/'collection.json')
    assert receipt['terminal'] and receipt['input_error'] is None and receipt['payload'] == pin(base/'payload.json')
    assert state['complete'] and state['code'] == receipt['code'] == (0 if all(c == 0 for c in codes) else 1)
    assert state['supervisor'] == read(base/'deployment.json') and state['boot_time'] == 1789634288.0
    assert payload['jobs'] == jobs == [r['name'] for r in state['runs']] and payload['limits'] == LIMITS
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    assert state['ended']-state['started'] < 4*3600
    for name,wanted in receipt['files'].items(): assert pin(folder/name) == wanted,name
    result = []
    for row, code in zip(state['runs'],codes,strict=True):
        assert row['complete'] and row['code'] == code and row['seconds'] < LIMITS['seconds']
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        samples = [json.loads(s) for s in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])] == m['birth'] for m in sample['members'])
        gaps = [samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0 <= gap < 10 for gap in gaps)
        result.append(dict(name=row['name'],code=code,samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    return result


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    original_resources = resources(BUILD,['sdk-version','cli-restore','cli-build','inventory'],[0,0,0,-6])
    recovered_resources = resources(BASE,['inventory'],[0])
    assert (BUILD/'collected/logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    folder = BASE/'collected'; payload = read(BASE/'payload.json'); stage = read(BASE/'bundle/stage.json')
    for name,wanted in stage['files'].items(): assert payload['files'][name] == wanted and pin(BASE/'bundle'/name) == wanted,name
    for name,row in stage['links'].items(): assert payload['files'][name] == row['identity'] and pin(folder/name) == row['identity'],name
    built = read(folder/'built.json'); original = read(BUILD/'collected/built.json')
    assert built['passed'] and built['product'] == original['product']
    for name,wanted in built['product'].items():
        assert pin(folder/'runtime'/name) == pin(BUILD/'collected/runtime'/name) == wanted
        assert original['files']['source/src/Lokad.Onnx.CLI/bin/Release/net10.0/'+name] == wanted
    il = inventory(read(folder/'inventory/instructions.json'),payload['measured'],built['product'],read(folder/'evidence/prior-composition.json'))
    assert il == read(folder/'inventory/review.json')
    analysis = dict(passed=True, measured=payload['measured'], built=built['product'], inventory=il,
        original_collection=pin(BUILD/'collected/collection.json'), original_resources=original_resources,
        recovered_resources=recovered_resources, original_inspection_failed=True,
        correction='Use complete qualified current-product dependency directory; Core/Data/Bridge/checks unchanged.',
        source_prepared=pin(BUILD/'collected/evidence/source-prepared.json'),
        no_rebuild=True,root_product_changed=False,numerically_qualified=False,no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=True,files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        original_collection=analysis['original_collection'],original_build_prepared=pin(BUILD/'prepared.json'),
        local_inputs=spec['files'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__ == '__main__': main()
