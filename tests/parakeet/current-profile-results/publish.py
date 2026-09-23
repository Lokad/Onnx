"""Publish the complete closed diagnostic, retaining every clock and stack weight."""
import csv,hashlib,json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-current-profile-amd-20260923'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def csv_file(name,rows):
    with (OUT/name).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
    assert not (OUT/'results-20260923.md').exists()
    proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(ROOT/name)==wanted,name
    value=read(BASE/'analysis.json');assert value['passed'] and value['calls']==240 and value['measured_calls']==180
    assert value['exact_prior_amd_results'] and proof['analysis']==pin(BASE/'analysis.json')
    clocks=[];setup=[];corpus=[];clips=[]
    for process in ['control','sampled-a','sampled-b']:
        result=read(BASE/'collected'/process/'result.json');assert len(result['records'])==80
        setup.append(dict(process=process,seconds=result['setup_seconds']))
        for row in result['records']:
            clocks.append(dict(process=process,**{k:row[k] for k in ['name','pass','phase','start_ticks','end_ticks','frequency','seconds','cpu_user_ticks','cpu_system_ticks','cpu_frequency','allocated_bytes','thread_id']}))
        selected=[r for r in result['records'] if r['phase']=='measured'];assert len(selected)==60
        wall=sum((Fraction(r['end_ticks']-r['start_ticks'],r['frequency']) for r in selected),Fraction())/3
        cpu=sum((Fraction(r['cpu_user_ticks']+r['cpu_system_ticks'],r['cpu_frequency']) for r in selected),Fraction())/3
        corpus.append(dict(process=process,wall_seconds=float(wall),wall_fraction=str(wall),cpu_seconds=float(cpu),allocated_bytes=sum(r['allocated_bytes'] for r in selected)/3))
        for name in dict.fromkeys(r['name'] for r in selected):
            rows=[r for r in selected if r['name']==name];assert len(rows)==3
            clips.append(dict(process=process,name=name,seconds=float(sum((Fraction(r['end_ticks']-r['start_ticks'],r['frequency']) for r in rows),Fraction())/3)))
    csv_file('clocks-20260923.csv',clocks);csv_file('setup-20260923.csv',setup);csv_file('clips-20260923.csv',clips)
    stacks=[];leaves={}
    for capture in value['diagnostics']:
        total=capture['selected_seconds']['corpus'];methods=defaultdict(float)
        for kind in ['exclusive','inclusive']:
            for row in capture[kind]:
                assert row['marker']=='corpus'
                stacks.append(dict(capture=capture['name'],kind=kind,method=row['method'],bucket=row.get('bucket',''),seconds=row['seconds'],share=row['seconds']/total))
                if kind=='exclusive':methods[row['method']]+=row['seconds']
        assert abs(sum(methods.values())/total-1)<1e-6
        leaves[capture['name']]={method:seconds/total for method,seconds in methods.items()}
    csv_file('stacks-20260923.csv',stacks)
    ranked=sorted(set().union(*(set(r) for r in leaves.values())),key=lambda m:sum(r.get(m,0) for r in leaves.values()),reverse=True)
    lines=['# Current Parakeet complete-request attribution','',
        'AMD EPYC9V74, CPU2, .NET10.0.8, unchanged release Core `521bae17` /',
        'Data `f3b9aa81`. Both complete-corpus captures pass the independent audit.',
        'These are sampled request-thread weights; inlining and collection overhead',
        'affect attribution. Wall and process CPU clocks are measured separately.',
        'No ORT worker is timed here and no product speedup is selected.','',
        '| Exclusive sampled leaf, full twenty-clip corpus | Capture A | Capture B |','|---|---:|---:|']
    for method in ranked[:16]:
        label=method.split('!',1)[-1].split('(',1)[0].removeprefix('Lokad.Onnx.').replace('|','\\|')
        lines.append(f"| {label} | {100*leaves['sampled-a'].get(method,0):.3f}% | {100*leaves['sampled-b'].get(method,0):.3f}% |")
    lines+=['','Ranked by combined share across both captures. [Every exclusive and inclusive',
        'stack weight](stacks-20260923.csv) is retained. Inclusive entries include',
        'their children and must not be added to exclusive percentages.','',
        '| Process | Corpus wall seconds | Process CPU seconds | Allocated bytes per corpus | Wall / control |','|---|---:|---:|---:|---:|']
    for row in corpus:
        lines.append(f"| {row['process']} | {row['wall_seconds']:.9f} | {row['cpu_seconds']:.9f} | {row['allocated_bytes']:.3f} | {row['wall_seconds']/corpus[0]['wall_seconds']:.6f} |")
    resources=value['resources'];samples=sum(r['samples'] for r in resources);peak=max(r['peak_rss'] for r in resources)
    lines+=['','Each process runs the same20clips /213.265seconds of audio, one warmup and',
        'three measured passes: **240complete public results,60warmups,180measurements**.',
        'The table sums twenty means per corpus; no clock is trimmed. [Every wall/CPU',
        'clock](clocks-20260923.csv), [setup](setup-20260923.csv) and [clip mean](clips-20260923.csv)',
        'is retained. All original native/public fields, current exact outputs, input',
        'hashes and held-output ownership checks pass.','',
        'Both Speedscope and Chromium exports reconcile completely. Every measured',
        'marker belongs to the original request thread; warmup markers are absent.',
        'Each full-corpus sampled duration meets the prospective5% wall-coverage bound.',
        'The exports expose54/55contiguous marker intervals for60measured calls each.',
        'Call counts come from complete request records, not interval counts; coverage',
        'and stack attribution are aggregate corpus checks, not per-clip attribution.',
        f'All **{samples:,} resource observations** pass; peak owned RSS **{peak:,}bytes**.',
        'Every build, capture and conversion owner is terminal. Collectors and monitors',
        'use CPU0. The copied product is unchanged;159of160existing consumer methods',
        'are exact, Main changes only family/wrapper dispatch and two wrappers are added.',
        '[Complete consumer instruction review](consumer-review-20260923.json) preserves',
        'the original public call, locals, exception regions and other branch targets.','',
        'The qualified matched baseline remains **76.172826seconds versus Microsoft',
        'ORT39.782592, ratio1.914728**. Diagnostic clocks do not replace that comparison.',
        'Use these observations to qualify a separate complete-call and application',
        'optimization. Parakeet is first priority; Pyannote follows, Whisper is deferred.','',
        'Artifact: `artifacts/parakeet-current-profile-amd-20260923`.',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.',
        'Payload: `'+pin(BASE/'payload/payload.json')['sha256']+'`.','']
    (OUT/'results-20260923.md').write_text('\n'.join(lines),encoding='utf8')
    report=dict(closed=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),generator=pin(Path(__file__)),corpus=corpus,resources=resources,
        diagnostics=[{k:v for k,v in d.items() if k not in ['exclusive','inclusive']} for d in value['diagnostics']],
        outputs={n:pin(OUT/n) for n in ['results-20260923.md','clocks-20260923.csv','setup-20260923.csv','clips-20260923.csv','stacks-20260923.csv']})
    (OUT/'observations-20260923.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,calls=len(clocks),resources=samples,peak_rss=peak,report=report['outputs']['results-20260923.md'])))
if __name__=='__main__':main()
