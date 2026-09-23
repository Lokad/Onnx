"""Publish every full-application clock, all three engines and the fixed verdict."""
import csv,json
from pathlib import Path
from protocol import TIMING_ROLES,pin,read
from checks import evaluate
from run import BASE

def main():
    proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    value=read(BASE/'analysis.json');assert pin(BASE/'analysis.json')==proof['analysis']
    assert evaluate(value['table'])==value['performance']
    out=Path(__file__).resolve().parent
    names=['results-20260923.md','clocks-20260923.csv','setup-20260923.csv','observations-20260923.json']
    assert not any((out/name).exists() for name in names)
    clocks=[];setups=[]
    for process,role in enumerate(TIMING_ROLES):
        result=read(BASE/'collected'/f'timing-{process:02}-{role}'/'output/result.json')
        setups.append(dict(process=process,role=role,seconds=result['setup_seconds']))
        clocks.extend(dict(process=process,role=role,**{k:r[k] for k in ['name','pass','phase','start_ticks','end_ticks','frequency','seconds']}) for r in result['records'])
    assert len(clocks)==480 and sum(r['phase']=='measured' for r in clocks)==360
    for name,rows in [('clocks-20260923.csv',clocks),('setup-20260923.csv',setups)]:
        with (out/name).open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    corpus=next(r for r in value['table'] if r['is_corpus']);performance=value['performance']
    lines=['# Parakeet first-use kernels versus current release and Microsoft ORT','',
        '**Application admitted; cross-model and root/package qualification remain required.**' if performance['admitted'] else '**Application rejected. This candidate is not integrated.**','',
        f"The twenty-clip corpus takes {corpus['current']['seconds']:.6f} seconds for current, {corpus['candidate']['seconds']:.6f} for candidate and {corpus['ort']['seconds']:.6f} for Microsoft ORT.",
        f"Candidate/ORT is {corpus['ratios_to_ort']['candidate']:.6f}. The <=1.05 parity target {'is met' if performance['parity_target_met'] else 'remains unmet'}.",'',
        '| Clip | Audio s | Current s | Candidate s | Microsoft ORT s | Candidate / ORT |',
        '|---|---:|---:|---:|---:|---:|']
    for row in value['table']:
        label='**Complete corpus**' if row['is_corpus'] else row['name']
        lines.append(f"| {label} | {row['audio_seconds']:.3f} | {row['current']['seconds']:.6f} | {row['candidate']['seconds']:.6f} | {row['ort']['seconds']:.6f} | {row['ratios_to_ort']['candidate']:.6f} |")
    lines+=['',f"Repeatability: {sum(r['passed'] for r in performance['controls'])}/63 controls pass. Admission gates: {sum(r['passed'] for r in performance['gates'])}/21 pass.",'',
        'AMD EPYC 9V74, CPU 2 before runtime startup, monitoring CPU 0; .NET 10.0.8',
        'and ORT 1.29.0 CPUExecutionProvider. ORT uses one intra/inter-op thread,',
        'sequential execution, all graph optimizations and no spinning. No profiler',
        'or numerical override is enabled. Products, consumers and native libraries',
        'are pinned to the recorded payload.','',
        'Six fresh processes run current, candidate, ORT, ORT, candidate, current.',
        'Each runs one warmup and three measured passes over all twenty clips.',
        'All 480 requests, 360 measurements and six setup intervals are retained.',
        'Corpus means sum the clip means within each process, then weight the two',
        'processes equally using exact integer-clock fractions. No samples are',
        'trimmed, pooled from earlier campaigns or retried.','',
        'Admission requires at least 3% lower corpus latency, no clip over 5% slower,',
        'and max/min of process means <=1.10 for each engine corpus and <=1.20 for',
        'every engine/clip. Every measured call includes frontend, neural inference,',
        'decoding and owned public results. Model setup and external validation are',
        'separate. Managed outputs match the freshly qualified reference exactly;',
        'the existing native transcript, token, input and ownership checks pass.','',
        f"All six workers are terminal. {sum(r['samples'] for r in value['resources']):,} resource observations pass; peak owned RSS {max(r['peak_rss'] for r in value['resources']):,} bytes.",
        'Foreign-CPU snapshot checks are retained, with their limitation that',
        'snapshots may miss short-lived work. Repeatability is checked separately.','',
        '[Every clock](clocks-20260923.csv), [setup intervals](setup-20260923.csv),',
        '[all controls, identities and results](observations-20260923.json).','',
        'Artifact: artifacts/parakeet-first-use-kernels-app-amd-20260923.',
        'Closure SHA256: '+pin(BASE/'closed.json')['sha256']+'.','']
    (out/'results-20260923.md').write_text('\n'.join(lines),encoding='utf8')
    observations=dict(closure=pin(BASE/'closed.json'),**value,clocks=pin(out/'clocks-20260923.csv'),setup=pin(out/'setup-20260923.csv'))
    (out/'observations-20260923.json').write_text(json.dumps(observations,indent=2)+'\n')
    print(json.dumps(dict(admitted=performance['admitted'],corpus=corpus,report=pin(out/'results-20260923.md'))))

if __name__=='__main__':main()
