"""Recompute report cells, public decisions, weight census and resource totals."""
from pathlib import Path
import importlib.util,json,sys
from common import ROOT,BASE,LOCAL,PRODUCT,PRELUDE,pin,read,write,ssh


def main():
    closed=read(BASE/'closed.json');assert closed['passed'] and closed['private_prototype'] and not closed['benchmark']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    ssh(PRELUDE+'terminal(%r)\n'%closed['births'])
    sys.path.insert(0,str(ROOT/'tests/whisper/memory-contracts-v2'))
    spec=importlib.util.spec_from_file_location('independent_recording_policy',ROOT/'tests/whisper/memory-contracts-v2/audit.py')
    policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    root=BASE/'collected';worker=root/'run/worker';value=read(worker/'result.json');weights=read(worker/'weights.json')
    native=read(root/'reference/native-recording.json');short=read(root/'short/manifest.json')
    folder=Path(__file__).parent;observations=read(folder/'observations-20260921.json');assert observations==read(BASE/'audit.json')
    report=(folder/'results-20260921.md').read_text(encoding='utf-8')
    for index,row in enumerate(value['cases']):
        assert policy.recording.decisions(row['result'])==policy.recording.decisions(native['cases'][index%4]['result'])
        recording=row['result'];name=row['name']+(' (repeat)' if row['repeat'] else '')
        display=f"| {name} | {len(recording['windows'])} | {len(recording['segments'])} | {recording['stop_reason']} | Yes |";assert display in report
    assert value['cases'][0]['result']==value['cases'][4]['result'] and value['short_recovery']==value['short_regression']
    policy.short_decisions(value['short_regression'],short['cases'][0])
    for row,case in zip(value['concurrent_speech'],short['cases'][:2],strict=True):policy.short_decisions(row['result'],case)
    overlap=(min(r['end'] for r in value['concurrent_speech'])-max(r['start'] for r in value['concurrent_speech']))/value['frequency']
    assert overlap>0 and overlap==observations['application']['overlap_seconds'] and f'**{overlap:.6f} seconds**' in report
    calls=len(value['cases'])+len(value['concurrent'])+len(value['concurrent_speech'])+4;assert calls==13 and value['refusals']==16
    policy.validate_sharing(dict(weight_sharing=weights));census=read(PRODUCT/'weight-census.json');checked=0
    for stage in ['before','after']:
        assert read(worker/('weights-'+stage+'.json'))==weights[stage]
        for name,graph in zip(['first','past'],census['graphs'],strict=True):
            entries={r['name']:r for r in weights[stage][name]['initializers']}
            for original in graph['rows']:
                current=entries[original['name']]
                assert current['shape']==original['shape'] and current['bytes']==original['bytes'] and current['sha256']==original['sha256']
                assert current['type']=={1:'Float',7:'Int64'}[original['type']];checked+=1
    assert checked==1136 and f'**{checked:,} original initializer comparisons**' in report
    samples=[json.loads(s) for s in (root/'run/samples.jsonl').read_text().splitlines()]
    peak=max(sum(m['rss'] for m in row['members']) for row in samples);available=min(row['available'] for row in samples)
    threads=sum(len(m['threads']) for s in samples for m in s['members']);resource=observations['resources']
    assert len(samples)==resource['samples'] and peak==resource['peak_rss'] and available==resource['min_available'] and threads==resource['thread_observations']
    assert f'**{peak:,} bytes**' in report and f'**{available:,} bytes**' in report
    assert f'**{len(samples):,} resource samples**' in report and f'**{threads:,} thread-affinity observations**' in report
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','WhisperMemoryContractsV2.dll']:assert pin(root/'bin'/name)==pin(LOCAL/'bin'/name)
    result=dict(passed=True,closure=pin(BASE/'closed.json'),pins=len(closed['files']),calls=calls,refusals=value['refusals'],
        displayed_rows=5,original_initializer_checks=checked,resource_totals_recomputed=True,overlap_verified=True,births=closed['births'])
    write(BASE/'final-verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
