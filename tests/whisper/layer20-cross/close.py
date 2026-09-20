"""Independently recompute reported pairs, retain all data, and close the cut."""
from pathlib import Path
import argparse,datetime,math,shutil
import numpy as np
from common import ROOT,pin,read,write,verify
from audit import original

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve()
    assert not (base/'closed.json').exists();spec=read(base/'manifest.json');verify(spec);audit=read(base/'audit.json')
    assert audit['passed'] is True and (audit['arrays'],audit['saved_array_bytes'],audit['workers'],audit['calls'])==(384,4423680000,16,32)
    assert audit['manifest']==pin(base/'manifest.json') and len(audit['instrumentation'])==32 and len(audit['contrasts'])==8
    for name,want in audit['files'].items():assert pin(base/name)==want,name
    births={(p['pid'],p['birth']) for r in audit['resources'] for p in r['births']}
    assert all(original.absent(dict(pid=pid,birth=birth)) for pid,birth in births)
    # Separate raw-array calculation checks all metrics used in the report table.
    pairs=0;bridges=0
    for contrast in audit['contrasts']:
        index=contrast['request'];item=spec['requests'][index]
        jobs={j['engine']:j for j in spec['schedule'] if j['request']==index}
        for number,node in enumerate(contrast['outputs']):
            values={kind:np.fromfile(base/'outputs'/jobs['managed' if kind[0]=='M' else 'native']['id']/f'{kind}-{number:02}.f32',dtype='<f4').reshape(node['shape']).astype(np.float64) for kind in ['MM','MN','NM','NN']}
            for incoming,m,n in [('managed_incoming','MM','NM'),('native_incoming','MN','NN')]:
                delta=values[m]-values[n];scaled=np.abs(delta)/np.maximum(1,np.abs(values[n]));row=node['pairwise_native_reference'][incoming]
                assert row['values']==scaled.size and row['failed_values']==int(np.count_nonzero(scaled>1e-4))
                assert row['max_scaled']==float(scaled.max()) and row['max_abs']==float(np.abs(delta).max())
                assert row['maximum_index']==list(np.unravel_index(int(np.argmax(scaled)),scaled.shape))
                assert math.isclose(row['rms'],float(np.sqrt(np.mean(delta*delta))),rel_tol=1e-12,abs_tol=1e-15);pairs+=1
            if number==11:
                for kind in ['MM','NN']:
                    reference=original.source(item['baselines'][kind]).astype(np.float64);scaled=np.abs(values[kind]-reference)/np.maximum(1,np.abs(reference))
                    row=next(r for r in audit['instrumentation'] if r['request']==index and r['kind']==kind)
                    assert row['extraction_bridge'] is True and row['max_scaled']==float(scaled.max()) and row['failed_values']==int(np.count_nonzero(scaled>1e-4));bridges+=1
    assert (pairs,bridges)==(192,16)
    eligible=[r for r in audit['instrumentation'] if r['extraction_bridge']]
    assert audit['extraction_bridges_passed'] is all(r['failed_values']==0 for r in eligible)
    write(base/'independent-verification.json',dict(passed=True,pairs=pairs,bridges=bridges,audit=pin(base/'audit.json'),births=[dict(pid=p,birth=b) for p,b in sorted(births)]))
    directory=ROOT/'tests/whisper/layer20-cross';report=directory/'results-20260920.md';observations=directory/'observations-20260920.json'
    assert not report.exists() and not observations.exists()
    write(observations,{k:v for k,v in audit.items() if k!='files'})
    lines=['# Natural Whisper layer-20 crossed-input diagnostic — September 20, 2026','',
        'All sixteen workers, thirty-two calls and 384 arrays pass execution, shape, identity, input preservation, held-output, repeat and resource checks. '
        'The existing cut contains exactly the original layer\'s 48 nodes and 27 initializers, with identical external weights. '
        'This diagnostic uses the three [selected natural cases](../trace-selected/results-20260920.md) and a repetition of the first, with both original feature sources.','',
        '**Extraction bridges '+('pass' if audit['extraction_bridges_passed'] else 'fail')+' the unchanged 1e-4 limit.** '+
        ('Each engine\'s cut on its own saved layer-19 input agrees with its traced layer-20 output within the declared limit. Actual differences remain below; agreement need not be bitwise.' if audit['extraction_bridges_passed'] else 'Some cut outputs differ from their same-engine traced layer-20 outputs beyond the declared limit. These extraction effects prevent attributing the original full-trace divergence solely to a cut operation.'),'',
        '| Case | Features | Engine | Final bits unchanged | Extraction max scaled | Failed values |',
        '|---|---|---|---|---:|---:|']
    for r in eligible:lines.append(f"| {r['name']}{' (repeat)' if r['request']>=6 else ''} | {r['features']} | {'managed' if r['kind']=='MM' else 'native'} | {r['bitwise']} | {r['max_scaled']:.9g} | {r['failed_values']:,} |")
    lines+=['','Each bridge uses its saved same-engine traced output as denominator. Both saved incoming arrays are separately run through both engines. '
        'In the cell names, the first letter is the cut engine and the second is the incoming-array producer: M for managed, N for native. '
        'The feature source is a separate dimension; it is not encoded in those two letters.','',
        '| Case | Features | First failure on managed incoming | First failure on native incoming | Final error on managed incoming | Final error on native incoming |',
        '|---|---|---|---|---:|---:|']
    for c in audit['contrasts']:
        if c['selected_request']==3:continue
        first=[]
        for incoming in ['managed_incoming','native_incoming']:
            failures=[n['name'] for n in c['outputs'] if n['pairwise_native_reference'][incoming]['failed_values']>0];first.append(failures[0] if failures else 'none')
        final=c['outputs'][-1]['pairwise_native_reference']
        lines.append(f"| {c['name']} | {c['features']} | {first[0]} | {first[1]} | {final['managed_incoming']['max_scaled']:.9g} | {final['native_incoming']['max_scaled']:.9g} |")
    lines+=['','These same-input comparisons use the corresponding native cut output as denominator. '
        '[Complete observations](observations-20260920.json) retain all twelve intermediate outputs, both within-engine incoming-input effects, '
        'six decomposition terms with a single NN denominator, absolute/scaled maxima and coordinates, RMS/L2 and every failing-value count. '
        'The two sums reconstruct the diagonal difference independently. Common-denominator terms are labeled separately from pairwise native-reference errors. '
        'No frame, padded value, case or failed array is excluded. Neither FP32 engine is established as mathematical truth.','',
        'Windows i7-14700KF, CPU2 inherited before startup, original core c6bf781/.NET10.0.12. Managed execution uses fresh Memory contexts and a256-MiB packing budget without native ORT. '
        'ORT1.29.0 has one intra/inter-op thread, sequential execution, all graph optimizations and spinning disabled. '
        'Actual first-call output objects stay held through subsequent calls/resets. Every repeated output matches its first-request bits. '
        'Native optimized graphs and external weights are retained, without running those serialized graphs on another host.','',
        f"Optimized node counts: managed {sum(audit['censuses']['managed'].values())}; native {sum(audit['censuses']['native'].values())}. A matching census alone does not establish matching arithmetic.",'',
        '| Engine | Resource samples | Peak sampled group RSS bytes | Minimum available bytes | Longest worker seconds |',
        '|---|---:|---:|---:|---:|']
    for engine in ['managed','native']:
        rows=[r for r in audit['resources'] if r['job']['engine']==engine]
        lines.append(f"| {engine} | {sum(r['samples'] for r in rows):,} | {max(r['peak_rss'] for r in rows):,} | {min(r['minimum_available'] for r in rows):,} | {max(r['seconds'] for r in rows):.3f} |")
    lines+=['','All workers remain within the fixed180-second/8-GiB sampled group-RSS/1-GiB available-memory limits, with10GiB available before each launch. '
        'Every original PID/creation-time identity is terminal. Sampling cannot establish an absolute peak. This is not a latency or full-corpus numerical qualification.','',
        f"Source `{spec['source_revision']}`; manifest SHA256 `{pin(base/'manifest.json')['sha256']}`; audit SHA256 `{pin(base/'audit.json')['sha256']}`.",
        'All4,423,680,000 raw output bytes, original source bindings, native optimized graphs, resource samples and receipts are retained under '
        '`artifacts/whisper-layer20-cross-20260920`. Product code and the original numerical gate remain unchanged.','']
    with report.open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    snapshot=base/'closed-source';snapshot.mkdir()
    for path in directory.iterdir():
        if path.is_file():shutil.copyfile(path,snapshot/path.name)
    files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    write(base/'closed.json',dict(passed=True,extraction_bridges_passed=audit['extraction_bridges_passed'],closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
        reports={p.relative_to(ROOT).as_posix():pin(p) for p in [report,observations]},births=[dict(pid=p,birth=b) for p,b in sorted(births)],scope=audit['scope']))
    print('Closed',len(files),'files;',pin(base/'closed.json'))

if __name__=='__main__':main()
