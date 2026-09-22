"""Audit diagnostics independently; retain each emitted tier and exact code text."""
import json
import re
from pathlib import Path
import run

c=run.c


def listings(path):
    text=path.read_text(encoding='utf8')
    # Raw logs remain authoritative if different compiler threads interleave.
    starts=list(re.finditer(r'; Assembly listing for method (.+) \(([^\n]+)\)\n',text))
    result=[]
    for i,start in enumerate(starts):
        end=starts[i+1].start() if i+1<len(starts) else len(text)
        body=text[start.start():end]
        sizes=re.findall(r'; Total bytes of code (\d+)',body)
        blocks=list(re.finditer(r'^(G_M\d+_IG\d+):[^\n]*\n',body,re.M))
        narrow=[]
        for j,b in enumerate(blocks):
            part=body[b.start():blocks[j+1].start() if j+1<len(blocks) else len(body)]
            if len(re.findall(r'\bvmulps\b',part))==3 and len(re.findall(r'\bvaddps\b',part))==3 and 'vpmaskmovd' in part:
                following=body[blocks[j+1].start():blocks[j+2].start() if j+2<len(blocks) else len(body)] if j+1<len(blocks) else ''
                narrow.append(dict(label=b.group(1),body=part,following=following))
        result.append(dict(method=start.group(1),tier=start.group(2),line=text.count('\n',0,start.start())+1,
            code_bytes=[int(s) for s in sizes],complete_uninterleaved=len(sizes)==1,
            narrow=narrow,body=body))
    return result


def main():
    assert not (c.BASE / 'closed.json').exists()
    prep=c.prepared(); spec=c.read(c.BASE / 'payload/payload.json'); collected=c.BASE / 'collected'
    receipt=c.read(collected / 'collection.json'); state=c.read(collected / 'identity.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for name,pin in receipt['files'].items():assert c.pin(collected / name)==pin
    assert receipt['payload']==prep['payload']==c.pin(collected / 'payload.json')
    transfer=c.read(c.BASE / 'collection-transfer.json')
    assert transfer['archive']==c.pin(c.BASE / 'results.tar.gz') and transfer['receipt']==c.pin(collected / 'collection.json')
    assert state['complete'] and state['code']==0 and state['boot_time']==spec['boot_time']
    assert [r['name'] for r in state['runs']]==spec['jobs']
    identities=[state['supervisor']]+[r['processes']['target'] for r in state['runs']]
    assert identities==receipt['identities'] or identities==[{k:receipt['identities'][0][k] for k in ['pid','birth']}]+receipt['identities'][1:]
    import transport
    confirmation=json.loads(transport.ssh(transport.PRELUDE+f'\nids={identities!r}\nassert not any(live(i) for i in ids)\nprint(json.dumps(dict(terminal=True)))\n'))
    assert confirmation['terminal']
    local=c.read(c.BASE / 'preparation.json'); assert local['complete'] and local['code']==0
    assert [r['name'] for r in local['runs']]==['restore','build','local-ordinary']
    c.terminal(local['supervisor']); resources=[]
    for location,rows,folder in [('Windows',local['runs'],c.BASE),('AMD',state['runs'],collected)]:
        for row in rows:
            assert row['complete'] and row['code']==0 and row['seconds']<900
            assert row['preflight']['available']>=8*1024**3
            if location=='AMD':assert row['preflight']['tmpfs']>=3*1024**3
            else:
                for pid,birth in row['members'].items():c.terminal(dict(pid=int(pid),birth=birth))
                assert row['preflight']==row['preflight_observations'][-1]
                assert all(s['seconds']<900 and s['disk']>=20*1024**3 for s in row['preflight_observations'])
            samples=[json.loads(line) for line in (folder / 'logs' / (row['name']+'.samples.jsonl')).read_text().splitlines()]
            assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
            for s in samples:
                assert s['seconds']<900 and s['rss']<(4 if row['name'] in ['restore','build'] else 2)*1024**3
                assert s['available']>=1024**3 and s['rss']==sum(p['rss'] for p in s['members'])
                if location=='AMD':
                    assert s['tmpfs']>=1024**3 and s['artifacts']<=1024**3 and s['monitor_affinity']==[0]
                    for p in s['members']:
                        assert {k:p[k] for k in ['pid','birth']}==row['processes']['target']
                        assert p['affinity']==[2] and p['threads'] and all(t['affinity']==[2] for t in p['threads'])
                else:
                    assert s['disk']>=20*1024**3 and s['output_bytes']<=1024**3
                    for p in s['members']:assert p['affinity']==[2] and row['members'][str(p['pid'])]==p['birth']
            resources.append(dict(location=location,name=row['name'],samples=len(samples),peak_rss=row['peak_rss']))
    reports=[]; code_dir=c.BASE / 'disassembly'; code_dir.mkdir()
    for name,job in [('local-ordinary',spec['job_details']['ordinary']),*spec['job_details'].items()]:
        is_local=name=='local-ordinary'; folder=c.BASE if is_local else collected
        result=c.read(folder / 'output' / (name+'.json')); manifest=c.read(c.BASE / 'payload' / job['shapes'])
        assert result['passed'] and result['core']==c.CORE and result['probe']==job['probe_sha256']
        assert result['executable']==c.pin(c.BASE / 'payload/runtime/TailCodegen.dll')['sha256']
        assert result['shapes']==c.pin(c.BASE / 'payload' / job['shapes'])['sha256'] and result['mode']==job['role']
        assert result['flags']==['DOTNET_JitDisasm'] and result['processor_count']==1 and result['fma'] and result['avx2']
        assert result['runtime']==('10.0.12' if is_local else '10.0.8')
        if not is_local: assert result['avx512']
        owner=next(r for r in (local if is_local else state)['runs'] if r['name']==name)
        assert result['pid']==(owner['worker'] if is_local else owner['processes']['target'])['pid']
        shapes=manifest['shapes']; assert len(shapes)==22
        assert [(r['phase'],r['m'],r['n'],r['k'],r['stride'],r['start']) for r in result['conditioning']]==[
            (phase,s['m'],s['n'],s['k'],s['stride'],s['timing_start']) for phase in range(2) for s in shapes]
        assert all(r['calls']>=16 and r['calls']%16==0 and r['seconds']>=1 for r in result['conditioning'])
        assert [{k:r[k] for k in ['m','n','k','stride','start','bias','pattern']} for r in result['validation']]==manifest['cases']
        assert all(r['passed'] and r['values']==r['m']*r['k'] and r['checked_buffer_values']==r['m']*r['stride']+6 for r in result['validation'])
        prior_key='ordinary' if 'ordinary' in name else 'masked'
        prior=c.ROOT / 'artifacts' / run.PRIORS[prior_key][0]
        old=c.read(prior / ('output/validate-local.json' if is_local else 'collected/output/validate.json'))['records']
        assert result['validation']==old
        digests={(r['m'],r['n'],r['k'],r['stride'],r['start']):r['digest'] for r in old if r['pattern']=='finite' and r['bias']}
        assert len(result['blocks'])==132
        for row,(s,block) in zip(result['blocks'],[(s,b) for s in shapes for b in range(6)],strict=True):
            assert row==dict(m=s['m'],n=s['n'],k=s['k'],stride=s['stride'],start=s['timing_start'],block=block,
                iterations=s['iterations'],digest=digests[(s['m'],s['n'],s['k'],s['stride'],s['timing_start'])])
        versions=listings(folder / 'logs' / (name+'.log'))
        wanted='Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed' if name=='baseline' else 'DirectOutput:Multiply'
        selected=[v for v in versions if v['method'].startswith(wanted+'(') and v['tier']=='Tier1']
        assert len(selected)==1 and selected[0]['complete_uninterleaved'] and len(selected[0]['narrow'])==1,(name,selected)
        for i,v in enumerate(versions):
            (code_dir / f'{name}-{i:02d}.asm').write_text(v.pop('body'),encoding='utf8')
        reports.append(dict(name=name,validation_cases=len(old),versions=versions,selected=selected[0]))
    analysis=dict(passed=True,diagnostic_only=True,resources=resources,remote_identities=identities,reports=reports)
    c.save(c.BASE / 'analysis.json',analysis)
    pins=dict(prep['files'])
    for folder in [collected,c.BASE / 'logs',c.BASE / 'output',code_dir,run.TOOLS]:
        pins.update({c.rel(p):c.pin(p) for p in folder.rglob('*') if p.is_file()})
    for p in c.BASE.iterdir():
        if p.is_file():pins[c.rel(p)]=c.pin(p)
    c.save(c.BASE / 'closed.json',dict(passed=True,files=pins,analysis=c.pin(c.BASE / 'analysis.json'),remote_identities=identities))
    print(json.dumps(dict(passed=True,resources=sum(r['samples'] for r in resources),closed=c.pin(c.BASE / 'closed.json'))))


if __name__=='__main__':main()
