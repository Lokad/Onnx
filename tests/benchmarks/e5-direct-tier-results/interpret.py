"""Summarize closed observations without changing clocks, gates or warmup policy."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-direct-tier-diagnostic-v2-amd'))
from protocol import pin,read
BASE=ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925'


def main():
    assert not (OUT/'interpretation-20260925.json').exists()
    assert pin(BASE/'closed.json')['sha256']=='e61941d9308e33b688f222665a0f79dd985f87b85d56123554a47202d8c531bc'
    assert pin(OUT/'observations-20260925.json')['sha256']=='a462b39fdb7eb75d7c13dda8e95114361d07779ae3b853aab3b4b441775b7e12'
    proof=read(BASE/'closed.json');assert pin(BASE/'analysis.json')==proof['analysis']
    analysis=read(BASE/'analysis.json');published=read(OUT/'observations-20260925.json');rows=[]
    for role,report in published['reports'].items():
        calls=analysis['reports'][role]['calls'];assert len(calls)==6000
        assert len(report['blocks'])==200 and not report['unmatched']
        first_ms=calls[0]['begin_ms'];window=calls[600:780]
        tiers=[]
        for method in report['matrix_timelines']:
            final,=[r for r in method['timeline'] if r['tier']=='OptimizedTier1']
            assert final['ms']>calls[779]['end_ms']
            tiers.append(dict(method=method['method'],method_id=method['method_id'],signature=method['signature'],
                final_seconds=final['since_first_call_ms']/1000,final_call=final['inside_call'],
                preceding_call=final['preceding_call'],final_native_bytes=final['bytes'],
                sizes_by_tier={r['tier']:r['bytes'] for r in method['timeline']}))
        # Summarize all ten consecutive 600-call groups; none is omitted or scored.
        groups=[dict(first=start,last=start+599,wall_ms=sum(c['wall_ms'] for c in calls[start:start+600])/600)
                for start in range(0,6000,600)]
        last=report['loads'][-1]
        rows.append(dict(process=role,product=report['product'],original_label_mean_ms=sum(c['wall_ms'] for c in window)/180,
            original_prefix_seconds=(calls[779]['end_ms']-first_ms)/1000,tiers=tiers,all_600_call_groups=groups,
            last_product_load=dict(last,since_first_call_seconds=(last['ms']-first_ms)/1000)))
    pairs={product:[r for r in rows if r['product']==product] for product in ['current','candidate']}
    spread={product:max(r['original_label_mean_ms'] for r in pair)/min(r['original_label_mean_ms'] for r in pair) for product,pair in pairs.items()}
    result=dict(diagnostic_only=True,new_inference=False,release_admitted=False,warmup_policy_changed=False,
        retrospective_cause_established=False,all_original_labels_and_extensions_retained=True,
        inputs={str(p.relative_to(ROOT)):pin(p) for p in [Path(__file__),BASE/'closed.json',BASE/'analysis.json',OUT/'observations-20260925.json']},
        original_label_process_spread=spread,rows=rows)
    with (OUT/'interpretation-20260925.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    lines=['# The exact candidate also outlives the short-e5 timing window','',
        'All four exact-product observations place the final optimized versions of',
        'both matrix dispatchers after call 779. The original timing labels therefore',
        'observe an intermediate runtime state in this diagnostic. This confirms the',
        'lead on the actual candidate; it does not explain the older 7.77% regression.','',
        '| Process | Product | Original 600–779 labels, ms | Matrix final tier, s / call | Batched final tier, s / call |',
        '|---|---|---:|---|---|']
    for row in rows:
        descriptions=[]
        for t in row['tiers']:
            at=str(t['final_call']) if t['final_call']>=0 else 'between '+str(t['preceding_call'])+'–'+str(t['preceding_call']+1)
            descriptions.append(f"{t['final_seconds']:.6f} / {at}")
        lines.append(f"| {row['process']} | {row['product']} | {row['original_label_mean_ms']:.6f} | "+' | '.join(descriptions)+' |')
    lines+=['','Seconds are elapsed from the first call. The two candidate processes differ',
        f"by {(spread['candidate']-1)*100:.2f}% in their original measurement labels; the release pair differs",
        f"by {(spread['current']-1)*100:.2f}%. These are descriptive instrumented clocks, not new performance gates.",
        'The full fixed 30-call table shows latency changes around and after final',
        'tier loads. Other product methods continue loading afterward; simultaneous',
        'compilation and runtime changes prevent attributing the change to one method.','',
        'All ten consecutive 600-call groups below provide a compact view of the',
        'entire observation. This post-capture grouping omits no call and is not a',
        'replacement for the predeclared 30-call blocks or a selected tail score.','',
        '| Calls | Release A, ms | Candidate B, ms | Candidate C, ms | Release D, ms |',
        '|---|---:|---:|---:|---:|']
    for index in range(10):
        group=[r['all_600_call_groups'][index] for r in rows]
        lines.append(f"| {group[0]['first']}–{group[0]['last']} | "+' | '.join(f"{r['wall_ms']:.6f}" for r in group)+' |')
    lines+=['','The runtime also reports different final native-code sizes across fresh',
        'processes using the same product and method. This is a concrete remaining',
        'lead; the event stream does not contain those machine instructions.','',
        '| Process | Matrix dispatcher final bytes | Batched dispatcher final bytes |',
        '|---|---:|---:|']
    for row in rows:lines.append(f"| {row['process']} | "+' | '.join(str(t['final_native_bytes']) for t in row['tiers'])+' |')
    lines+=['','Do not infer identical optimized code from a shared tier label. Conversely,',
        'code size alone neither identifies the differing instructions nor proves',
        'their latency cost. Source-level kernel changes, a JIT-setting sweep or',
        'a favorable rerun would be premature. The next bounded diagnosis should',
        'inspect the actual emitted code for these two exact method identities and',
        'their call routes, preserving runtime settings and the complete call history.',
        'First establish whether retained evidence already contains usable code bytes;',
        'prepare a capture only if that read-only inspection establishes a specific gap.','',
        'All original graph failures remain failed. Release admission, the original',
        'warmup policy and BENCHMARK.md stay unchanged. Separate shared/Pyannote',
        'correctness qualification remains available while that diagnosis proceeds.','',
        '[Complete report and every 30-call block](report-20260925.md),',
        '[exact interpretation inputs and summaries](interpretation-20260925.json).','',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    with (OUT/'interpretation-20260925.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(interpretation=pin(OUT/'interpretation-20260925.json'),spread=spread,
        native_sizes={r['process']:[t['final_native_bytes'] for t in r['tiers']] for r in rows})))


if __name__=='__main__':main()
