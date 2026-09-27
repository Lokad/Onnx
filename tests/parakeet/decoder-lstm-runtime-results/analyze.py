"""Attribute the retained diagnostic trace without replaying or rescoring any run."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAPTURE = ROOT/'artifacts/parakeet-decoder-lstm-runtime-observation-amd-20260927'
OUT = ROOT/'artifacts/parakeet-decoder-lstm-runtime-analysis-20260927'


def read(path): return json.loads(path.read_text())
def pin(path):
    with path.open('rb') as stream: return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
def save(path, value):
    with path.open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def union(segments):
    result = []
    for begin, end in sorted(segments):
        assert begin <= end
        if result and begin <= result[-1][1]: result[-1][1] = max(result[-1][1], end)
        else: result.append([begin, end])
    return result


def overlap(segments, calls):
    return sum(max(0., min(end, call['end_ms'])-max(begin, call['begin_ms']))
        for begin, end in segments for call in calls)


def main():
    assert not OUT.exists()
    assert pin(CAPTURE/'closed.json')['sha256'] == 'a00f9195ab9be512b1230357d68e3e1142593014a79680198cc5d517e95b6c85'
    proof = read(CAPTURE/'closed.json'); assert proof['passed'] and proof['diagnostic_only']
    for name, wanted in proof['files'].items(): assert pin(CAPTURE/name) == wanted, name
    joined = read(CAPTURE/'intervals.json'); calls = joined['intervals']
    result = read(CAPTURE/'collected/trace-capture/output/result.json')
    events = [json.loads(s) for s in (CAPTURE/'collected/trace-export/events/events.jsonl').read_text().splitlines()]
    pending = defaultdict(list); jit = []; collections = {}; gc = []; suspend = {}; pauses = []
    for event in events:
        if event['provider'] != 'Microsoft-Windows-DotNETRuntime': continue
        payload = event['payload']; name = event['name']; thread = event['thread']
        if name == 'Method/JittingStarted': pending[(thread, payload['MethodID'])].append(event)
        elif name == 'Method/LoadVerbose':
            prior = pending[(thread, payload['MethodID'])].pop()
            for key in ['MethodID', 'ModuleID', 'MethodNamespace', 'MethodName', 'MethodSignature']:
                assert prior['payload'][key] == payload[key], key
            jit.append(dict(begin_ms=prior['ms'], end_ms=event['ms'], thread=thread,
                elapsed_ms=event['ms']-prior['ms'], start_event=prior['index'], stop_event=event['index'],
                method_id=payload['MethodID'], module_id=payload['ModuleID'],
                method=payload['MethodNamespace']+'::'+payload['MethodName'],
                signature=payload['MethodSignature'], tier=payload['OptimizationTier']))
        elif name == 'GC/Start':
            assert payload['Count'] not in collections; collections[payload['Count']] = event
        elif name == 'GC/Stop':
            prior = collections.pop(payload['Count'])
            assert prior['payload']['Depth'] == payload['Depth']
            gc.append(dict(count=int(payload['Count']), begin_ms=prior['ms'], end_ms=event['ms'],
                elapsed_ms=event['ms']-prior['ms'], kind=prior['payload']['Type'], reason=prior['payload']['Reason'],
                depth=int(payload['Depth']), start_thread=prior['thread'], stop_thread=thread,
                start_event=prior['index'], stop_event=event['index']))
        elif name == 'GC/SuspendEEStart':
            assert thread not in suspend; suspend[thread] = dict(start=event)
        elif name == 'GC/SuspendEEStop': suspend[thread]['suspended'] = event
        elif name == 'GC/RestartEEStart': suspend[thread]['restart'] = event
        elif name == 'GC/RestartEEStop':
            prior = suspend.pop(thread); start = prior['start']; stopped = prior['suspended']; restart = prior['restart']
            assert start['ms'] <= stopped['ms'] <= restart['ms'] <= event['ms']
            pauses.append(dict(reason=start['payload']['Reason'], thread=thread,
                request_ms=start['ms'], suspended_ms=stopped['ms'], restart_ms=restart['ms'], complete_ms=event['ms'],
                start_event=start['index'], stop_event=event['index']))
    assert not any(pending.values()) and not collections and not suspend
    assert len(jit) == 1532 and len(gc) == 21 and len(pauses) == 9287
    assert all(j['elapsed_ms'] >= 0 for j in jit)
    threads = sorted({j['thread'] for j in jit}); worker = result['native_thread']
    state = read(CAPTURE/'collected/identity.json'); capture = state['runs'][4]
    samples = [json.loads(s) for s in (CAPTURE/'collected/logs/trace-capture.jsonl').read_text().splitlines()]
    seen_threads = {t['tid'] for s in samples for p in s['members'] if p['pid'] == result['pid']
        for t in p['threads'] if t['affinity'] == [2]}
    assert set(threads) <= seen_threads
    phases = []
    for phase in ['warmup', 'measured']:
        for repeat in range(5):
            selected = [r for r in calls if r['phase'] == phase and r['repeat'] == repeat]
            jit_by_thread = [dict(thread=thread, worker_thread=thread == worker,
                union_elapsed_overlap_ms=overlap(union((j['begin_ms'], j['end_ms']) for j in jit if j['thread'] == thread), selected)) for thread in threads]
            gc_by_kind = {kind: overlap(union((g['begin_ms'], g['end_ms']) for g in gc if g['kind'] == kind), selected)
                for kind in sorted({g['kind'] for g in gc})}
            suspension_by_reason = {reason: overlap(union((p['suspended_ms'], p['restart_ms']) for p in pauses if p['reason'] == reason), selected)
                for reason in sorted({p['reason'] for p in pauses})}
            phases.append(dict(phase=phase, repeat=repeat, begin_ms=selected[0]['begin_ms'], end_ms=selected[-1]['end_ms'],
                calls=len(selected), call_elapsed_ms=sum(r['elapsed_ms'] for r in selected),
                gc_counter_pause_delta_ms=sum(r['pause_delta_ms'] for r in selected),
                jit_by_thread=jit_by_thread, gc_elapsed_overlap_ms=gc_by_kind,
                suspended_overlap_ms=suspension_by_reason))
    slow = max((r for r in phases if r['phase'] == 'measured'), key=lambda r:r['call_elapsed_ms'])
    assert slow['repeat'] == 0
    active = [j for j in jit if j['begin_ms'] < slow['end_ms'] and j['end_ms'] > slow['begin_ms']]
    assert len(active) == 569
    osr = [j for j in active if j['method'] == 'Lokad.Onnx.CPUExecutionProvider::Lstm' and j['tier'] == 'OptimizedTier1OSR']
    assert len(osr) == 1 and osr[0]['thread'] == worker
    enclosing = [r for r in calls if r['begin_ms'] <= osr[0]['begin_ms'] and osr[0]['end_ms'] <= r['end_ms']]
    assert len(enclosing) == 1
    analysis = dict(passed=True, diagnostic_only=True, screen_rescored=False, capture_closure=pin(CAPTURE/'closed.json'),
        worker_pid=result['pid'], worker_thread=worker, product=read(CAPTURE/'payload.json')['product'],
        original_calls=3800, exact_output_arrays=11400, raw_events=len(events), lost=0,
        clock_offset_bounds_ms=joined['offset_bounds_ms'], clock_tolerance_ms=joined['clock_tolerance_ms'],
        jit_intervals=1532, gc_intervals=21, suspension_reasons=dict(Counter(p['reason'] for p in pauses)),
        phases=phases, slowest_measured_repeat=0, jit_intervals_overlapping_slow_phase=569,
        slow_phase_jit_tiers=dict(Counter(j['tier'] for j in active)),
        slow_phase_largest_jit_intervals=sorted(active, key=lambda j:j['elapsed_ms'], reverse=True)[:8],
        direct_lstm_osr_interval=osr[0], enclosing_lstm_call=enclosing[0],
        conclusion='The declared measured region includes substantial tiered compilation, including an LSTM OSR compilation inside one call. GC is not the primary observed disturbance. Tracing shifts the slow pass, so the original untraced pass cannot be assigned an exact cause or corrected retrospectively.',
        limits='JIT/GC spans and their overlaps are elapsed intervals, not CPU time or additive recoverable latency. SuspendOther includes profiler activity and is not a GC collection. Prepared-path between-process variability remains unresolved.',
        next_decision='Keep the layout candidate unchanged. Qualify complete Parakeet correctness, then judge its application benefit independently at the existing complete-transcription boundary; retain the failed component screen and its limitations.')
    OUT.mkdir(); save(OUT/'jit.json', jit); save(OUT/'gc.json', dict(collections=gc, suspensions=pauses))
    save(OUT/'analysis.json', analysis)
    save(OUT/'closed.json', dict(passed=True, diagnostic_only=True, analysis=pin(OUT/'analysis.json'),
        source=pin(Path(__file__)), files={p.name:pin(p) for p in OUT.iterdir() if p.is_file()}))
    target = Path(__file__).with_name('runtime-20260927.json')
    save(target, dict(**analysis, analysis_closure=pin(OUT/'closed.json')))
    print(json.dumps(dict(passed=True, closure=pin(OUT/'closed.json'), report=pin(target), slow_phase=slow,
        direct_lstm_osr_ms=osr[0]['elapsed_ms'], untraced_screen_rescored=False)))


if __name__ == '__main__': main()
