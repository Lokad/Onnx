"""Independent raw-record arithmetic, application checks and final receipt verification."""
import json,math
from deploy import BASE,ROOT,ssh
from protocol import pin,read,write,LIMITS


def main():
    closed=read(BASE/'closed.json');assert closed['passed'] and not closed['normal_runtime_qualification'] and not closed['benchmark']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    base=BASE/'collected';manifest=read(base/'manifests/whisper.json');audit=read(BASE/'audit.json');state=read(base/'campaign/identity.json')
    run=state['runs'][0];folder=base/run['output'];value=read(folder/'worker/result.json')
    assert len(value['records'])==len(manifest['cases'])==20
    for index,(row,case) in enumerate(zip(value['records'],manifest['cases'],strict=True)):
        assert read(folder/'worker'/f'{index:03}.json')==row
        assert row['result']==case['expected'] and row['input_sha256']==case['raw_sha256'] and row['name']==case['name']
        assert row['ownership'] is True
    report=(ROOT/'tests/whisper/memory-collection/results-20260920.md').read_text(encoding='utf-8')
    for n,(c,summary) in enumerate(zip(value['collections'],audit['interventions'],strict=True)):
        assert c['after_call']==[8,16,20][n] and read(folder/'worker'/f"collection-{c['after_call']:02}.json")==c
        a=c['before'];b=c['after'];assert b['collections'][2]>a['collections'][2] and b['last_gc_index']>a['last_gc_index']
        assert b['managed_estimate']==a['managed_estimate']-summary['managed_reclaimed']
        assert b['rss']==a['rss']-summary['rss_reduced']
        assert math.isclose(c['seconds'],(c['end_ticks']-c['start_ticks'])/c['frequency'],rel_tol=1e-14)
        assert c['held_outputs_unchanged'] and c['inputs_unchanged']
        after_call=value['records'][c['after_call']-1]
        assert after_call['end_ticks']<=a['ticks']<=c['start_ticks']<c['end_ticks']<=b['ticks']
        if c['after_call']<20:assert b['ticks']<=value['records'][c['after_call']]['memory_before']['ticks']
        line=f"| {c['after_call']} | {a['managed_estimate']/1e9:.6f} | {b['managed_estimate']/1e9:.6f} | {(a['managed_estimate']-b['managed_estimate'])/1e9:.6f} | {a['rss']/1e9:.6f} | {b['rss']/1e9:.6f} | {c['seconds']*1000:.3f} |"
        assert line in report
    samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
    assert len(samples)==audit['resource']['samples']==run['samples']
    assert max(sum(p['rss'] for p in s['members']) for s in samples)==audit['resource']['peak_rss']
    assert min(s['available'] for s in samples)==audit['resource']['min_available']>=LIMITS['available']
    assert all(s['disk']>=LIMITS['disk'] and sum(p['rss'] for p in s['members'])<LIMITS['rss'] for s in samples)
    assert all(p['affinity']==[2] and all(t['affinity']==[2] for t in p['threads']) for s in samples for p in s['members'])
    result=ssh('''import sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
print('terminal')
'''%closed['births']);assert result.strip()=='terminal'
    write(BASE/'final-verification.json',dict(passed=True,closure=pin(BASE/'closed.json'),files=len(closed['files']),calls=20,
        interventions=3,displayed_cells=21,resource_samples=len(samples),births=closed['births']))
    print(json.dumps(read(BASE/'final-verification.json')))


if __name__=='__main__':main()
