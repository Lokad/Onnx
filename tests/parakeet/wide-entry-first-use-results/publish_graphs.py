"""Publish every clock and original gate from the closed wide-entry graph campaign."""
import csv,json,shutil,sys
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/parakeet/wide-entry-first-use-graphs-amd'))
from protocol import ORDER,pin,read
from statistics import summarize
BASE=ROOT/'artifacts/parakeet-wide-entry-first-use-graphs-amd-20260923'

def write(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:f.write(value)

def main():
    proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(BASE/'payload.json')['sha256']=='685ccd420597c5a6563c724489cf60a9021d7c82f61e92153f8330fce5249b90'
    a=read(BASE/'analysis.json');assert a['clocks']==37512 and a['measured']==8640
    assert a['consumer']['branches_locals_exceptions_equal'] and a['consumer']['implementation_flags_equal']
    for r in a['performance']:
        assert {k:v for k,v in r.items() if k!='key'}==summarize({role:read(BASE/'collected'/('timing-'+r['key']+'-'+role)/'output/result.json') for role in ORDER})
    target=OUT/'graphs-clocks-20260923.csv';assert not target.exists();shutil.copyfile(BASE/'clocks.csv',target)
    setups=[];blocks=[]
    for row in read(BASE/'collected/identity.json')['runs']:
        name=row['name']
        if not name.startswith(('verify-','timing-')):continue
        v=read(BASE/'collected'/name/'output/result.json');setups.append(dict(process=name,seconds=v['setup_seconds']))
        if name.startswith('timing-'):
            for start in [600,660,720]:
                clocks=v['clocks'][start:start+60];assert len(clocks)==60 and all(not c['warmup'] for c in clocks)
                mean=sum((Fraction(c['ticks'],c['frequency']) for c in clocks),Fraction())/60
                blocks.append(dict(process=name,first=start,last=start+59,seconds=float(mean),numerator=mean.numerator,denominator=mean.denominator))
    assert len(setups)==72 and len(blocks)==144
    for name,rows in [('graphs-setup-20260923.csv',setups),('graphs-blocks-20260923.csv',blocks)]:
        with (OUT/name).open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write('graphs-observations-20260923.json',json.dumps(dict(closure=pin(BASE/'closed.json'),payload=pin(BASE/'payload.json'),**a),indent=2,allow_nan=False)+'\n')
    verdict='**All warmed-workload release checks pass.**' if proof['admitted'] else '**Warmed-workload release checks do not all pass; the candidate remains unselected.**'
    lines=['# Wide-entry Parakeet candidate: complete graph comparison','',verdict,'',
        '| Case | Selected ms | Candidate ms | Microsoft ORT ms | Candidate / ORT | Candidate / selected |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in a['performance']:
        lines.append(f"| {r['key']} | {r['current']*1000:.6f} | {r['candidate']*1000:.6f} | {r['ort']*1000:.6f} | {r['ratio']:.6f} | {r['candidate_over_current']:.6f} |")
    lines+=['','AMD EPYC 9V74 CPU 2, monitoring CPU 0; .NET 10.0.8, ORT 1.29.0 CPU.',
        'Lokad measures Reset, Execute and owned output arrays; ORT measures',
        'session.run and its returned arrays. Setup, input creation and validation',
        'are separate. Native settings remain one intra/inter-op thread, sequential',
        'execution and all graph optimizations. No profiler or runtime overrides.','',
        'All 24 numerical processes run first, three calls each. Each case then uses',
        'six fresh timing processes in selected, candidate, ORT, ORT, candidate,',
        'selected order: 600 fixed warmups and 180 measurements per process.',
        'All 37,512 clocks, 8,640 measurements, 72 setups and 144 measured blocks',
        'are retained. Exact rational clocks give equal process weights. No trimming,',
        'favorable block selection, pooling with the predecessor or unchanged retry.','',
        f"Repeatability: {sum(c['passed'] for r in a['performance'] for c in r['controls'])}/24 controls pass (same-engine process max/min <=1.10).",
        f"Regression: {sum(r['regression_passed'] for r in a['performance'])}/8 checks pass (candidate/selected <=1.05).",'',
        'Every candidate output matches selected bytes. All products satisfy fresh',
        'ORT scaled error <=1e-4, exact shapes, finiteness, unchanged inputs and',
        'held-output ownership. All clocks remain even when a control fails.','',
        f"Consumer inspection covers all {a['consumer']['methods']} methods, with {a['consumer']['unchanged_methods']} unchanged.",
        'Main changes only call count 120 to 780 and warmup boundary 60 to 600.',
        'Every other instruction, branch target, local, exception region, method',
        'flag and public declaration is preserved. Both products use that consumer;',
        'the native runner changes exactly the same two constants.','',
        'The compiled warmed consumer is reused unchanged from its retained',
        'instruction/flag qualification. No build or diagnostic worker overlaps',
        'these measurements. The candidate preserves the original shared entry',
        'and arithmetic paths; its new dispatch is limited to eligible wide',
        'matrices. Admission still requires every declared graph check.', '',
        f"All owners are terminal; {sum(r['samples'] for r in a['resources']):,} resource observations pass. Peak RSS {max(r['peak_rss'] for r in a['resources']):,} bytes.",'',
        '[All clocks](graphs-clocks-20260923.csv), [fixed measured blocks](graphs-blocks-20260923.csv),',
        '[all setups](graphs-setup-20260923.csv), [consumer, controls and resources](graphs-observations-20260923.json).','',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    write('graphs-20260923.md','\n'.join(lines)+'\n')
    print(json.dumps(dict(admitted=proof['admitted'],clocks=a['clocks'],measured=a['measured'],blocks=len(blocks))))

if __name__=='__main__':main()
