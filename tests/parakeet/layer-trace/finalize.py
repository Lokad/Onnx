"""Independently verify retained data and render the complete diagnostic report."""
from common import *
import math
import numpy as np

DYNAMIC = ROOT/'artifacts/parakeet-layer-trace-dynamic-20260921'
INSPECT = ROOT/'artifacts/parakeet-trace-optimization-20260921'
FAILED = ROOT/'artifacts/parakeet-layer-trace-20260921'
OUT = ROOT/'artifacts/parakeet-layer-trace-final-20260921'
REPORT = Path(__file__).with_name('results-20260921.md')
OBSERVATIONS = Path(__file__).with_name('observations-20260921.json')


def main():
    assert not OUT.exists() and not REPORT.exists() and not OBSERVATIONS.exists()
    spec=read(BASE/'manifest.json');verify(spec)
    old=read(BASE/'analysis.json');new=read(DYNAMIC/'analysis.json')
    assert not old['trace_qualified'] and new['qualified'] and len(new['comparisons'])==78
    bound={}
    for base in (BASE,DYNAMIC):
        receipt=read(base/'closed.json')
        for name,expected in receipt['files'].items():
            assert pin(ROOT/name)==expected,name;bound[name]=expected
        bound[rel(base/'closed.json')]=pin(base/'closed.json')
    for name,expected in spec['files'].items():
        assert pin(ROOT/name)==expected,name;bound[name]=expected
    for name,expected in read(DYNAMIC/'prepared.json')['files'].items():
        assert pin(ROOT/name)==expected,name;bound[name]=expected
    for name,expected in read(FAILED/'preparation-failure.json')['files'].items():
        path=FAILED/name;assert pin(path)==expected;bound[rel(path)]=expected
    bound[rel(FAILED/'preparation-failure.json')]=pin(FAILED/'preparation-failure.json')
    identities=[];samples=0;peak=0
    for base,state_name in ((BASE,'processes.json'),(DYNAMIC,'state.json'),(INSPECT,'state.json')):
        state=read(base/state_name);assert state['complete'] and state['code']==0
        identities.append(state['supervisor'])
        for row in state['runs']:
            assert row['complete'] and row['code']==0;identities.append(row['worker'])
            raw=row['samples'] if isinstance(row['samples'],list) else [json.loads(s) for s in (base/'process'/row['job']['id']/'samples.jsonl').read_text().splitlines()]
            for s in raw:
                assert s['seconds']<LIMITS['seconds'] and s['rss']<LIMITS['rss'] and s['available']>=LIMITS['available'] and s['disk']>=LIMITS['disk'] and s['affinity']==[2]
            samples+=len(raw);peak=max(peak,max(s['rss'] for s in raw))
    assert len(identities)==22 and all(absent(i) for i in identities)
    # Re-read complete arrays, independently checking max/count, L2 with fsum,
    # and the retained argmax. No reported scalar is accepted on its own.
    verified_values=0
    for row in new['comparisons']:
        kind='native' if row['route']=='native-features' else 'managed'
        native_kind='managed' if row['route']=='managed-features' else 'native'
        folder=BASE/'outputs'/('managed-'+kind+'-trace')
        record=next(r for r in read(folder/'result.json')['outputs'] if r['name']==row['name'])
        actual=array(folder/record['file'],record)
        folder=DYNAMIC/'outputs'/native_kind
        record=next(r for r in read(folder/'result.json')['outputs'] if r['name']==row['name'])
        reference=array(folder/record['file'],record)
        assert actual.shape==reference.shape and actual.size==row['values']
        differences=[float(a)-float(b) for a,b in zip(actual.reshape(-1),reference.reshape(-1),strict=True)]
        errors=[abs(d)/max(1.,abs(float(b))) for d,b in zip(differences,reference.reshape(-1),strict=True)]
        assert max(errors)==row['max_scaled'] and sum(e>1e-4 for e in errors)==row['failures']
        assert errors.index(max(errors))==row['worst_index']
        assert math.isclose(math.sqrt(math.fsum(d*d for d in differences)),row['l2'],rel_tol=2e-14,abs_tol=1e-15)
        verified_values+=actual.size
    assert verified_values==5910528
    observations=dict(original=old,corrected=new,optimization=read(INSPECT/'differences.json'),
                      original_closure=pin(BASE/'closed.json'),corrected_closure=pin(DYNAMIC/'closed.json'))
    write(OBSERVATIONS,observations)
    lines=['# Parakeet Windows encoder diagnostic — 2026-09-21','',
      'The corrected trace preserves both engines’ original computation on the selected',
      'English clip. It exposes numerical differences at the subsampling stem and late',
      'encoder layers that the final encoder output alone conceals. The full Windows',
      'duration-logit failure remains unresolved; no production arithmetic or tolerance changes.', '',
      'The fresh unmodified-model controls also complete the two-by-two input substitution:',
      'the same native decoder call uses frame 49, target 1627 and the original incoming',
      'recurrent states from accepted step 25. Only its encoder values change.', '',
      '| Feature producer | Encoder engine | Duration-zero logit | Maximum scaled decoder error | Values above 1e-4 |',
      '|---|---|---:|---:|---:|']
    for job in ('native-native-plain','managed-native-plain','native-managed-plain','managed-managed-plain'):
        row=next(r for r in old['decoder'] if r['job']==job and r['name']=='outputs')
        engine,feature,_=job.split('-')
        lines.append(f'| {feature} | {engine} | {row["duration_zero"]:.12g} | {row["max_scaled"]:.9g} | {row["failures"]} |')
    lines += ['',
      'Native features reduce the managed-encoder/fixed-decoder discrepancy to',
      '6.091594696e-5 on this call. With managed features it is 2.268552780e-4.',
      'The original complete managed trajectory remains at 2.321004868e-4; this',
      'experiment deliberately fixes the decoder’s incoming states. The two feature',
      'substitutions have different effects in the two encoders, so their contributions',
      'cannot be treated as a simple additive error budget. No independent higher-precision',
      'reference establishes either FP32 engine as the mathematical truth.', '',
      '## Complete boundary comparisons','',
      'Each cell is the maximum scaled error, followed by the number of values exceeding',
      '`abs(actual-reference)/max(1,abs(reference)) <= 1e-4`. The first two columns feed',
      'identical features to both encoders. The third uses each engine’s own retained',
      'features. All 75,776 values per float boundary are checked. Layer indices are zero-based.', '',
      '| Boundary | Common native features | Common managed features | Each engine’s features |',
      '|---|---:|---:|---:|']
    names=[n for n in spec['outputs']['trace'][2:]]+['outputs']
    for name in names:
        label='Stem projection' if name.startswith('/pre_encode') else 'Final encoder output' if name=='outputs' else 'Layer '+name.split('/')[1].split('.')[1]
        cells=[]
        for route in ('native-features','managed-features','natural-inputs'):
            row=next(r for r in new['comparisons'] if r['route']==route and r['name']==name)
            cells.append(f'{row["max_scaled"]:.9g} ({row["failures"]})')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['',
      'The final encoder tensor is a transpose of layer 23. Its small errors do not',
      'qualify every intermediate: the stem already fails on both identical-input routes,',
      'and errors rise again before the last layer. Under natural inputs, layer 22 reaches',
      '8.155703545e-4 with 307 failing values, then the final output falls to 6.839632988e-6.',
      'These are accumulated differences, not proof that a particular layer is incorrect.', '',
      '## Instrumentation failure and correction','',
      'The first preparation stops before inference because it assumes the wrong native',
      'package directory. Its clean build and all partial assets/source remain preserved.',
      'The corrected preparation binds the actual loaded ORT libraries.', '',
      'The first inference schedule completes all 12 encoder calls and 12 fixed-state',
      'decoder calls, but fails two native original/trace controls. Added outputs were',
      'given fixed shapes `[1,74,1024]`; native optimization then specializes dynamic',
      'shape calculations. Separate load-only inspections retain 1,993 original versus',
      '1,350 instrumented native nodes, with 643 changed or removed nodes. That trace',
      'cannot support attribution to the original native computation. Every failed',
      'control and its raw arrays remain in the original closure.', '',
      'The separately frozen correction gives the new outputs three unknown dimensions.',
      'Four new native calls cover both feature inputs and repeats. It reuses eight valid',
      'original controls/managed captures. All 59 correction checks pass: original final',
      'output bits, every repeated output, and the complete original native optimized',
      'node inventory including attribute hashes. The managed original/trace plans each',
      'contain 2,856 identical nodes. All three historical unmodified-model controls',
      'match their retained outputs exactly.', '',
      '## Evidence and limits','',
      f'The combined record contains 28 inference calls, 380 raw output arrays, {samples}',
      f'resource samples and 22 verified terminal process identities. Two additional',
      'session-construction calls serialize optimized graphs without running inference.',
      f'The largest sampled process RSS is {peak:,} bytes. The 8 GiB RSS, 1 GiB',
      'available-memory, 20 GiB disk and 900-second worker guards remain unchanged.',
      'All input/held-output checks and exact repeats pass. This is a numerical',
      'diagnostic on Windows i7-14700KF CPU2, not a latency comparison.', '',
      'The current qualified core hash is',
      '`'+CORE+'` (product `0f86c5d`), runtime 10.0.12; native ORT is 1.29.0.',
      'Both original models, external weights, retained input arrays, actual native',
      'libraries, capture consumer and diagnostic sources are byte-bound. The',
      '[complete observations](observations-20260921.json) include all 78 boundary',
      'comparisons, original failed controls, decoder comparisons and native graph differences.',
      'All 5,910,528 boundary value comparisons receive independent scalar max/count/L2 checks.', '',
      'Original inference closure: `'+pin(BASE/'closed.json')['sha256']+'`.',
      'Corrected trace closure: `'+pin(DYNAMIC/'closed.json')['sha256']+'`.',
      'The final read-only audit and report pins live in',
      '`artifacts/parakeet-layer-trace-final-20260921/verified.json`.', '',
      'The next distinct investigation is the subsampling stem on identical inputs,',
      'with independent higher-precision reference calculations before choosing a',
      'production arithmetic change. The selected English case and fixed decoder states',
      'do not establish full transcription, other-language or whole-model numerical acceptance.',
      'The existing Microsoft ORT timing baselines and all existing numerical failures remain.', '']
    REPORT.write_text('\n'.join(lines),encoding='utf8')
    OUT.mkdir();bound[rel(OBSERVATIONS)]=pin(OBSERVATIONS);bound[rel(REPORT)]=pin(REPORT);bound[rel(Path(__file__))]=pin(__file__)
    write(OUT/'verified.json',dict(files=bound,identities=identities,all_terminal=True,resource_samples=samples,peak_rss=peak,
                                 boundary_comparisons=78,independently_checked_values=verified_values,original_failure_retained=True,corrected_trace_passed=True))
    print(json.dumps(dict(files=len(bound),identities=len(identities),samples=samples,report=rel(REPORT),verification=pin(OUT/'verified.json'))))


if __name__=='__main__':
    main()
