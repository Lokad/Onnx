"""Independently reconstruct request totals, report cells and serialized model identities."""
from decimal import Decimal
from pathlib import Path
import json
from deploy import BASE,ROOT
from close import terminal
from protocol import pin,read,write


def main():
    closed=read(BASE/'closed.json');assert closed['passed'] and closed['prototype_only'] and not closed['benchmark']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    terminal(closed['births']);base=BASE/'collected';state=read(base/'campaign/identity.json')
    manifest=read(base/'manifests/whisper.json');audit=read(BASE/'audit.json');folder=Path(__file__).parent
    assert read(folder/'observations-20260920.json')==audit
    report=(folder/'results-20260920.md').read_text(encoding='utf-8');calls=0;initializer_checks=0
    census=read(BASE/'weight-census.json')
    expected=sum(r['bytes'] for r in census['graphs'][1]['rows'] if r.get('exact_match_in_first') and r['type']==1 and r['bytes']>=4096)
    assert expected==635187200
    for run,observation in zip(state['runs'],audit['observations'],strict=True):
        worker=base/run['output']/'worker';paths=sorted(worker.glob('[0-9][0-9][0-9].json'));assert len(paths)==observation['calls']
        values=[json.loads(p.read_text(),parse_float=Decimal) for p in paths]
        result=read(worker/'result.json');pair=result['weight_sharing'];assert pair['before']==pair['after']
        storage=pair['before'];assert storage['logical_shared_bytes']==expected<=storage['shared_payload_bytes']
        for name,graph in zip(['first','past'],census['graphs'],strict=True):
            initializers={r['name']:r for r in storage[name]['initializers']}
            for row in graph['rows']:
                current=initializers[row['name']]
                assert current['bytes']==row['bytes'] and current['sha256']==row['sha256'] and current['shape']==row['shape']
                assert current['type']=={1:'Float',7:'Int64'}[row['type']]
                initializer_checks+=1
        for i,row in enumerate(values):
            case=manifest['cases'][i%20];assert row['name']==case['name'] and row['pass']==i//20
            assert row['result']==case['expected'] and row['input_sha256']==case['raw_sha256'] and row['ownership'] is True
            if i:assert row['pools']['encodingExecution']['allocated_new_bytes']<=16777216
            for key,budget in [('encodingExecution',536870912),('firstExecution',134217728),('pastExecution',134217728)]:
                pool=row['pools'][key];assert pool['cache_budget']==budget and pool['cache_bytes']<=budget and pool['cache_count']<=256
            calls+=1
        allocations=[v['allocated_bytes'] for v in values];assert sum(allocations)==observation['allocated_total']
        samples=[json.loads(s) for s in (base/run['output']/'samples.jsonl').read_text().splitlines()]
        peak=max(sum(m['rss'] for m in s['members']) for s in samples);available=min(s['available'] for s in samples)
        assert peak==observation['resource']['peak_rss'] and available==observation['resource']['min_available']
        assert peak<15032385536 and available>=1073741824
        display=f"| {'Conformance' if run['phase']=='conformance' else 'Endurance'} | {len(values)} | {peak:,} | {available:,} | {storage['shared_payload_bytes']:,} |"
        assert display in report
        if run['phase']=='conformance':
            original=[read(base/'original-prefix'/f'{i:03}.json') for i in range(16)]
            before=sum(v['allocated_bytes'] for v in original[1:]);after=sum(v['allocated_bytes'] for v in values[1:16])
            assert after*2<=before and before==audit['gate']['original_allocated_bytes'] and after==audit['gate']['prototype_allocated_bytes']
            assert f'**{after:,} bytes**' in report and f'**{before:,} bytes**' in report
            assert f'({Decimal(after)/Decimal(before):.3%})' in report
    assert calls==100 and initializer_checks==1136
    source=read(BASE/'source.json')
    for name,wanted in source['files'].items():assert pin(BASE/'source'/name)==wanted,name
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:
        assert pin(BASE/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'/name)==pin(BASE/'bin'/name)==pin(base/'bin'/name)
    result=dict(passed=True,closure=pin(BASE/'closed.json'),pins=len(closed['files']),calls=calls,serialized_initializer_checks=initializer_checks,
        displayed_rows=2,logical_shared_bytes=expected,births=closed['births'])
    write(BASE/'final-verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
