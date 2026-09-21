"""Recompute published values from integer timer ticks and check the closed evidence."""
from decimal import Decimal,localcontext
import json,math,re
from common import *

def main():
    closed=read(BASE/'closed.json');assert closed['passed']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    base=BASE/'collected';state=read(base/'campaign/identity.json');audit=read(BASE/'audit.json')
    assert state['complete'] and state['code']==0 and len(state['runs'])==5
    manifest=read(base/'manifests/whisper.json');raw={};counts=dict(conformance=0,warmup=0,measured=0);samples_total=0
    for run in state['runs']:
        folder=base/run['output'];value=read(folder/'worker/result.json');raw[run['name']]=value
        assert len(value['records'])==(20 if run['phase']=='conformance' else 80)
        for i,row in enumerate(value['records']):
            assert read(folder/'worker'/f'{i:03}.json')==row
            assert row['name']==manifest['cases'][i%20]['name'] and row['pass']==i//20
            assert math.isclose(row['seconds'],(row['end_ticks']-row['start_ticks'])/row['frequency'],rel_tol=1e-14)
            counts['conformance' if run['phase']=='conformance' else row['phase']]+=1
        samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        assert len(samples)==run['samples'];peak=max(sum(m['rss'] for m in r['members']) for r in samples)
        assert peak==run['peak_rss'];observed=next(o for o in audit['observations'] if o['name']==run['name'])['resource']
        assert peak==observed['peak_rss'] and len(samples)==observed['samples']
        assert min(r['available'] for r in samples)==observed['min_available']
        assert min(r['disk'] for r in samples)==observed['min_disk'];samples_total+=len(samples)
        for row in samples:
            assert row['available']>=LIMITS['available'] and row['disk']>=LIMITS['disk'] and 0<=row['seconds']<LIMITS['seconds']
            assert sum(m['rss'] for m in row['members'])<LIMITS['rss']
            for m in row['members']:
                assert m['affinity']==[2] and m['birth']==run['members'][str(m['pid'])]
                assert m['threads'] and all(t['affinity']==[2] for t in m['threads'])
    assert counts==dict(conformance=20,warmup=80,measured=240)
    report=(ROOT/'tests/audio/whisper-amd/results-20260921.md').read_text(encoding='utf-8')
    benchmark=(ROOT/'BENCHMARK.md').read_text(encoding='utf-8');section=benchmark.split('### Audio: matched AMD Whisper baseline\n',1)[1].split('\n### ',1)[0]
    checks=0
    with localcontext() as ctx:
        ctx.prec=50
        for observed in audit['table']:
            cases=manifest['cases'] if observed['name']=='complete-corpus' else [c for c in manifest['cases'] if c['name']==observed['name']]
            assert cases;names={c['name'] for c in cases};duration=Decimal(sum(c['samples'] for c in cases))/Decimal(16000);values={}
            for engine in ['managed','ort']:
                selected=[raw[r['name']] for r in state['runs'] if r['phase']=='timing' and r['engine']==engine];assert len(selected)==2
                rows=[r for worker in selected for r in worker['records'] if r['pass']>0 and r['name'] in names];assert len(rows)==6*len(cases)
                seconds=sum((Decimal(r['end_ticks']-r['start_ticks'])/Decimal(r['frequency']) for r in rows),Decimal(0))/Decimal(6);values[engine]=seconds
                assert math.isclose(float(seconds),observed[engine]['seconds'],rel_tol=1e-14)
                assert math.isclose(float(seconds/duration),observed[engine]['rtf'],rel_tol=1e-14);checks+=2
                for visit in observed[engine]['visits']:
                    for p,total in zip([1,2,3],visit['passes'],strict=True):
                        expected=sum((Decimal(r['end_ticks']-r['start_ticks'])/Decimal(r['frequency']) for r in raw[visit['process']]['records'] if r['pass']==p and r['name'] in names),Decimal(0))
                        assert math.isclose(float(expected),total,rel_tol=1e-14);checks+=1
            ratio=values['managed']/values['ort'];assert math.isclose(float(ratio),observed['ratio'],rel_tol=1e-14);checks+=1
            suffix=' | '+' | '.join(f'{float(v):.3f}' for v in [values['managed'],values['ort'],ratio,values['managed']/duration,values['ort']/duration])+' |'
            assert suffix in report
            if observed['name']=='complete-corpus':assert suffix in section
    assert len(audit['table'])==21
    links=0
    for path in [ROOT/'tests/audio/whisper-amd/results-20260921.md',ROOT/'BENCHMARK.md',ROOT/'docs/model-support.md']:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):continue
            assert (path.parent/link.split('#')[0]).exists(),(path,link);links+=1
    ssh(PRELUDE+'terminal(%r)\n'%closed['births'])
    write(BASE/'final-verification.json',dict(passed=True,closure=pin(BASE/'closed.json'),pins=len(closed['files']),counts=counts,resource_samples=samples_total,decimal_checks=checks,table_rows=21,links=links,births=closed['births']))
    print(json.dumps(read(BASE/'final-verification.json')))

if __name__=='__main__':main()
