"""Retain the failed all-method capture without weakening its completeness gate."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/pyannote/winograd-output-blocks-codegen'))
from prepare import BASE,previous_closed
from protocol import JOBS,LIMITS,pin,read,save,check_sample
from numerical_checks import result,same_current,identity
from listings import listings


def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    prepared=read(BASE/'prepared.json')
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    c=BASE/'collected';receipt=read(c/'collection.json');state=read(c/'identity.json')
    payload=read(BASE/'payload.json');transfer=read(BASE/'collection-transfer.json');built=read(c/'built.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json')
    assert state['boot_time']==1789634288.0 and state['ended']-state['started']<4*3600
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS
    for name,wanted in built['files'].items():assert pin(c/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items():assert pin(c/'runtimes'/role/name)==wanted
    fixtures={(r['case'],r['index']):r for r in read(BASE/'bundle/evidence/fixtures.json')['calls'] if r['eligible'] and r['attributes']['strides']==[1,1]}
    rows=[];code={};reports={};failures={}
    for row in state['runs']:
        name=row['name'];role,mode,width_text=name.split('-');width=int(width_text)
        assert mode=='captured' and row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(line) for line in (c/'logs'/(name+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        value=read(c/name/'result.json');reference=read(BASE/'bundle/evidence'/(name+'.json'))
        identity(value,role,payload,built,row);same_current(value,reference)
        assert value['assembly']==reference['assembly'] and value['core_sha256']==reference['core_sha256']
        reports[name]=result(value,mode,width,fixtures);assert not reports[name]['failures']
        code[name]=listings(c/name/'jit.asm');assert code[name]
        failures[name]=[dict(method=b['method'],tier=b['tier'],line=b['line'],bytes=b['code_bytes'],
            raw_sha256=b['raw_sha256']) for b in code[name] if not b['complete_body'] or not b['complete_uninterleaved'] or b['managed_stdout_repairs']]
        actual={(b['method'].split(':')[1].split('(')[0],b['tier']) for b in failures[name]}
        expected={(f'OutputWinograd{width}','Instrumented Tier0'),(f'MultiplyWinograd{width}','Tier1-OSR')} if role=='current' else set()
        assert actual==expected and len(failures[name])==len(expected)
        for method in ['MultiplyWinograd'+width_text,'OutputWinograd'+width_text,'EpilogueRange']:
            assert any(':'+method+'(' in b['method'] and b['tier'].startswith('Tier1') for b in code[name])
        assert any(':TransformWinogradInputContiguous(' in b['method'] and (b['tier'].startswith('Tier1') or b['tier']=='Tier0-FullOpts') for b in code[name])
        rows.append(dict(name=name,samples=len(samples),peak_rss=row['peak_rss'],bodies=len(code[name])))
    save(BASE/'listings.json',code)
    analysis=dict(passed=False,retained_failure=True,codegen_admitted=False,numerical_checks_passed=True,resources_passed=True,
        resources=rows,failures=failures,products=payload['products'],consumer=built['consumer'],numerical_results=reports,
        no_performance_measurement=True,reason='Selected-product OutputWinograd instrumented output interleaves with MultiplyWinograd OSR output in both widths. Original completeness gate failed; no reconstructed or omitted tier is admitted.')
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=False,retained_failure=True,codegen_admitted=False,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},no_performance_measurement=True))
    out=Path(__file__).with_name('codegen-failure-observations-20260923.json')
    save(out,dict(closure=pin(BASE/'closed.json'),analysis=analysis,closer=pin(Path(__file__))))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),resources=rows,failures=failures)))


if __name__=='__main__':main()
