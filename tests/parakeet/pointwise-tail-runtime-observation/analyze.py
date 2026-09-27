"""Attribute only the closed diagnostic's clocks; never revise the timing screen."""
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAPTURE = ROOT/'artifacts/parakeet-pointwise-tail-runtime-observation-amd-20260927'
OUT = ROOT/'artifacts/parakeet-pointwise-tail-runtime-analysis-20260927'
source = ROOT/'tests/parakeet/decoder-lstm-runtime-results/analyze.py'
loader = importlib.util.spec_from_file_location('retained_runtime_analysis', source)
prior = importlib.util.module_from_spec(loader); loader.loader.exec_module(prior)
read, pin, save, union, overlap = prior.read, prior.pin, prior.save, prior.union, prior.overlap


def runtime_intervals(events):
    pending = defaultdict(list); jit = []; collections = {}; gc = []; suspend = {}; pauses = []
    for event in events:
        if event['provider'] != 'Microsoft-Windows-DotNETRuntime': continue
        payload = event['payload']; name = event['name']; thread = event['thread']
        if name == 'Method/JittingStarted': pending[(thread, payload['MethodID'])].append(event)
        elif name == 'Method/LoadVerbose':
            start = pending[(thread, payload['MethodID'])].pop()
            for key in ['MethodID', 'ModuleID', 'MethodNamespace', 'MethodName', 'MethodSignature']:
                assert start['payload'][key] == payload[key], key
            jit.append(dict(begin_ms=start['ms'], end_ms=event['ms'], thread=thread,
                elapsed_ms=event['ms']-start['ms'], start_event=start['index'], stop_event=event['index'],
                method_id=payload['MethodID'], module_id=payload['ModuleID'],
                method=payload['MethodNamespace']+'::'+payload['MethodName'],
                signature=payload['MethodSignature'], tier=payload['OptimizationTier']))
        elif name == 'GC/Start':
            assert payload['Count'] not in collections; collections[payload['Count']] = event
        elif name == 'GC/Stop':
            start = collections.pop(payload['Count']); assert start['payload']['Depth'] == payload['Depth']
            gc.append(dict(count=int(payload['Count']), begin_ms=start['ms'], end_ms=event['ms'],
                elapsed_ms=event['ms']-start['ms'], kind=start['payload']['Type'], reason=start['payload']['Reason'],
                depth=int(payload['Depth']), start_thread=start['thread'], stop_thread=thread,
                start_event=start['index'], stop_event=event['index']))
        elif name == 'GC/SuspendEEStart':
            assert thread not in suspend; suspend[thread] = dict(start=event)
        elif name == 'GC/SuspendEEStop': suspend[thread]['suspended'] = event
        elif name == 'GC/RestartEEStart': suspend[thread]['restart'] = event
        elif name == 'GC/RestartEEStop':
            pair = suspend.pop(thread); start = pair['start']; stopped = pair['suspended']; restart = pair['restart']
            assert start['ms'] <= stopped['ms'] <= restart['ms'] <= event['ms']
            pauses.append(dict(reason=start['payload']['Reason'], thread=thread,
                request_ms=start['ms'], suspended_ms=stopped['ms'], restart_ms=restart['ms'], complete_ms=event['ms'],
                start_event=start['index'], stop_event=event['index']))
    assert not any(pending.values()) and not collections and not suspend
    assert all(j['elapsed_ms'] >= 0 for j in jit)
    counts = Counter(e['name'] for e in events if e['provider'] == 'Microsoft-Windows-DotNETRuntime')
    assert len(jit) == counts['Method/JittingStarted'] == counts['Method/LoadVerbose']
    assert len(gc) == counts['GC/Start'] == counts['GC/Stop']
    assert len(pauses) == counts['GC/SuspendEEStart'] == counts['GC/SuspendEEStop'] == counts['GC/RestartEEStart'] == counts['GC/RestartEEStop']
    return jit, gc, pauses


def main():
    assert not OUT.exists()
    assert pin(CAPTURE/'closed.json')['sha256'] == '473104209f3a9161dc0600df123581fbb6fe626bea8894be69388b953277332e'
    proof = read(CAPTURE/'closed.json'); assert proof['passed'] and proof['diagnostic_only']
    for name, wanted in proof['files'].items(): assert pin(CAPTURE/name) == wanted, name
    joined = read(CAPTURE/'intervals.json'); calls = joined['intervals']; assert len(calls) == 400
    result = read(CAPTURE/'collected/trace-capture/output/result.json')
    events = [json.loads(s) for s in (CAPTURE/'collected/trace-export/events/events.jsonl').read_text().splitlines()]
    jit, gc, pauses = runtime_intervals(events)
    threads = sorted({j['thread'] for j in jit}); worker = result['native_thread']
    samples = [json.loads(s) for s in (CAPTURE/'collected/logs/trace-capture.jsonl').read_text().splitlines()]
    seen_threads = {t['tid'] for s in samples for p in s['members'] if p['pid'] == result['pid']
        for t in p['threads'] if t['affinity'] == [2]}
    assert set(threads) <= seen_threads
    jit_segments = {t:union((j['begin_ms'],j['end_ms']) for j in jit if j['thread'] == t) for t in threads}
    gc_segments = {kind:union((g['begin_ms'],g['end_ms']) for g in gc if g['kind'] == kind) for kind in sorted({g['kind'] for g in gc})}
    pause_segments = {reason:union((p['suspended_ms'],p['restart_ms']) for p in pauses if p['reason'] == reason)
        for reason in sorted({p['reason'] for p in pauses})}
    def attribution(selected):
        return dict(calls=len(selected), call_elapsed_ms=sum(r['elapsed_ms'] for r in selected),
            gc_counter_pause_delta_ms=sum(r['pause_delta_ms'] for r in selected),
            jit_by_thread=[dict(thread=t, worker_thread=t == worker, union_elapsed_overlap_ms=overlap(jit_segments[t],selected)) for t in threads],
            gc_elapsed_overlap_ms={k:overlap(v,selected) for k,v in gc_segments.items()},
            suspended_overlap_ms={k:overlap(v,selected) for k,v in pause_segments.items()})
    phases = []
    for phase in ['warmup','measured']:
        for repeat in range(5):
            selected = [r for r in calls if r['phase'] == phase and r['repeat'] == repeat]
            assert len(selected) == 40
            phases.append(dict(phase=phase, repeat=repeat, begin_ms=selected[0]['begin_ms'],
                end_ms=selected[-1]['end_ms'], **attribution(selected)))
    per_call = [dict(**r, **{k:v for k,v in attribution([r]).items() if k != 'calls'}) for r in calls]
    shapes = read(CAPTURE/'collected/spec.json')['shapes']; per_shape = []
    for index, shape in enumerate(shapes):
        selected = [r for r in calls if r['phase'] == 'measured' and r['call'] == index]
        times = [r['elapsed_ms'] for r in selected]; assert len(times) == 5
        per_shape.append(dict(index=index, **shape, times_ms=times, max_min_ratio=max(times)/min(times),
            slowest_repeat=max(selected,key=lambda r:r['elapsed_ms'])['repeat'], **attribution(selected)))
    measured = [r for r in calls if r['phase'] == 'measured']
    active = [dict(**j, elapsed_overlap_ms=overlap([(j['begin_ms'],j['end_ms'])],measured)) for j in jit]
    active = [j for j in active if j['elapsed_overlap_ms'] > 0]
    direct = [dict(j, enclosing_calls=[r['ordinal'] for r in calls if r['begin_ms'] <= j['begin_ms'] <= j['end_ms'] <= r['end_ms']])
        for j in active if j['thread'] == worker]
    analysis = dict(passed=True, diagnostic_only=True, screen_rescored=False,
        capture_closure=pin(CAPTURE/'closed.json'), worker_pid=result['pid'], worker_thread=worker,
        product=read(CAPTURE/'payload.json')['product'], original_calls=400, exact_output_arrays=400,
        raw_events=len(events), lost=0, clock_offset_bounds_ms=joined['offset_bounds_ms'],
        clock_tolerance_ms=joined['clock_tolerance_ms'], jit_intervals=len(jit), gc_intervals=len(gc),
        suspension_reasons=dict(Counter(p['reason'] for p in pauses)), phases=phases,
        measured_attribution=attribution(measured), shapes=per_shape,
        measured_jit_intervals=active, direct_measured_jit_intervals=direct,
        product_kernel_compilations=[j for j in jit if j['method'].startswith('Lokad.Onnx.MathOps::')],
        limits='Elapsed JIT/GC overlaps are not CPU time or additive recoverable latency. Sampling suspensions perturb this trace. Old untraced calls have no event attribution and remain uncorrected. No component or application speedup is admitted.')
    OUT.mkdir(); save(OUT/'jit.json', jit); save(OUT/'gc.json', dict(collections=gc,suspensions=pauses))
    save(OUT/'calls.json', per_call); save(OUT/'analysis.json', analysis)
    save(OUT/'closed.json', dict(passed=True, diagnostic_only=True, analysis=pin(OUT/'analysis.json'),
        source=pin(Path(__file__)), reused_analysis_source=pin(source),
        files={p.name:pin(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(dict(passed=True,closure=pin(OUT/'closed.json'),jit=len(jit),gc=len(gc),
        measured_attribution=analysis['measured_attribution'],
        unstable_shapes=sum(s['max_min_ratio']>1.10 for s in per_shape),
        direct_measured_jit=direct)))


if __name__ == '__main__': main()
