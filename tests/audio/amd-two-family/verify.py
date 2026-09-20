"""Independently recompute the published table with Decimal and verify closure."""
from decimal import Decimal,localcontext
import json,math,re
from stage import BASE,ROOT,ssh
from protocol import pin,read,write


def main():
    closed=read(BASE/'closed.json');assert closed['passed']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    base=BASE/'collected';state=read(base/'campaign/identity.json');audit=read(BASE/'audit.json')
    assert len(state['runs'])==8 and state['complete'] and state['code']==0
    raw={};calls=warmup=measured=0
    for run in state['runs']:
        result=read(base/run['output']/'worker/result.json');raw[run['name']]=result
        for i,row in enumerate(result['records']):
            assert read(base/run['output']/'worker'/f'{i:03}.json')==row
            calls+=1
            if row['pass']==0:assert row['phase']=='warmup';warmup+=1
            else:assert row['phase']=='measured' and row['pass'] in [1,2,3];measured+=1
    assert (calls,warmup,measured)==(384,96,288)
    report=(ROOT/'tests/audio/amd-two-family/results-20260920.md').read_text(encoding='utf-8')
    benchmark=(ROOT/'BENCHMARK.md').read_text(encoding='utf-8')
    section=benchmark.split('### Audio: matched AMD Parakeet and pyannote baselines\n',1)[1].split('\n### ',1)[0]
    checks=0;table_rows=0
    with localcontext() as ctx:
        ctx.prec=50
        for observed in audit['table']:
            manifest=read(base/'manifests'/(observed['family']+'.json'))
            cases=manifest['cases'] if observed['name']=='complete-corpus' else [c for c in manifest['cases'] if c['name']==observed['name']]
            assert cases;names={c['name'] for c in cases};duration=Decimal(sum(c['samples'] for c in cases))/Decimal(16000)
            values={}
            for engine in ['managed','ort']:
                selected=[r for r in raw.values() if r['family']==observed['family'] and r['engine']==engine]
                assert len(selected)==2
                rows=[r for worker in selected for r in worker['records'] if r['pass']>0 and r['name'] in names]
                assert len(rows)==6*len(cases)
                seconds=sum((Decimal(str(r['seconds'])) for r in rows),Decimal(0))/Decimal(6)
                values[engine]=seconds
                assert math.isclose(float(seconds),observed[engine]['seconds'],rel_tol=1e-14,abs_tol=1e-14)
                assert math.isclose(float(seconds/duration),observed[engine]['rtf'],rel_tol=1e-14,abs_tol=1e-14);checks+=2
                for visit in observed[engine]['visits']:
                    worker=raw[visit['process']];assert worker['family']==observed['family'] and worker['engine']==engine
                    for p,total in zip([1,2,3],visit['passes'],strict=True):
                        expected=sum((Decimal(str(r['seconds'])) for r in worker['records'] if r['pass']==p and r['name'] in names),Decimal(0))
                        assert math.isclose(float(expected),total,rel_tol=1e-14,abs_tol=1e-14);checks+=1
            ratio=values['managed']/values['ort'];assert math.isclose(float(ratio),observed['ratio'],rel_tol=1e-14);checks+=1
            cells=[values['managed'],values['ort'],ratio,values['managed']/duration,values['ort']/duration]
            suffix=' | '+' | '.join(f'{float(v):.3f}' for v in cells)+' |'
            assert suffix in report
            if observed['name']=='complete-corpus' or observed['family']=='pyannote':assert suffix in section;table_rows+=1
    assert table_rows==5 and len(audit['table'])==25
    links=0
    for path in [ROOT/'tests/audio/amd-two-family/results-20260920.md',ROOT/'BENCHMARK.md',ROOT/'docs/model-support.md']:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):continue
            assert (path.parent/link.split('#')[0]).exists(),(path,link);links+=1
    result=ssh('''import sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
print('terminal')
'''%closed['births']);assert result.strip()=='terminal'
    write(BASE/'final-verification.json',dict(passed=True,closure=pin(BASE/'closed.json'),files=len(closed['files']),calls=calls,warmup=warmup,measured=measured,
        decimal_checks=checks,table_rows=25,headline_rows=table_rows,links=links,terminal_births=closed['births']))
    print(json.dumps(read(BASE/'final-verification.json')))


if __name__=='__main__':main()
