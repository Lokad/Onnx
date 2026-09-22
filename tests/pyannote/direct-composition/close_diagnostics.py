"""Freeze intermediate and baseline-self diagnostics with terminal resources."""
import json
import successor

c=successor.configured()
for name,jobs in [('pyannote-direct-caller-diagnostic-20260922',['restore','build','diagnostic']),
                  ('pyannote-baseline-nan-20260922',['restore','build','block-1','block-7'])]:
    base=c['ROOT'] / 'artifacts' / name;assert not (base / 'closed.json').exists()
    state=c['read'](base / 'processes.json');assert state['complete'] and state['code']==0
    assert [r['name'] for r in state['runs']]==jobs
    identities=[state['supervisor']];count=0
    for row in state['runs']:
        expected=3762504530 if row['name']=='diagnostic' else 0
        assert row['complete'] and row['code']==expected and row['seconds']<900
        samples=[json.loads(s) for s in (base / 'logs' / (row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for s in samples:
            assert s['seconds']<900 and s['rss']<8*1024**3 and s['available']>=1024**3 and s['disk']>=20*1024**3 and s['output_bytes']<=1024**3
            assert s['rss']==sum(p['rss'] for p in s['members'])
            for p in s['members']:assert p['affinity']==[2] and row['members'][str(p['pid'])]==p['birth']
        identities.extend(dict(pid=int(pid),birth=birth) for pid,birth in row['members'].items());count+=len(samples)
    for i in identities:c['terminal'](i)
    inputs=c['read'](base / 'inputs.json');c['verify'](inputs['files'])
    if jobs[-1]=='diagnostic':
        result=c['read'](base / 'result.json.mismatch.json')
        assert result['index']==1720 and result['observed_old']=='7fc12345' and result['observed_new']=='ffc00000'
        assert result['old_raw']==result['current_raw']==result['direct_unbiased']=='ffc00000' and result['bias']=='7fc12345'
    else:
        one=c['read'](base / 'block-1.json');seven=c['read'](base / 'block-7.json')
        assert one['passed'] and seven['passed'] and one['differing_nan_payloads']==0 and seven['differing_nan_payloads']==6
        assert seven['differences'][0]==dict(index=1720,cold='ffc00000',hot='7fc12345')
        assert all(r['inputs_unchanged'] and r['all_non_nan_bits_unchanged'] and r['calls']>30 and r['seconds']>=2 for r in [one,seven])
        result=dict(block1=one,block7=seven)
    files=dict(inputs['files']);files.update({c['rel'](p):c['pin'](p) for p in base.rglob('*') if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(base).parts)})
    c['save'](base / 'closed.json',dict(passed=True,diagnostic_only=True,files=files,identities=identities,resource_samples=count,result=result))
    print(json.dumps(dict(name=name,closed=c['pin'](base / 'closed.json'),resources=count)))
