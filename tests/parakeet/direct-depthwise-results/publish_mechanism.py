"""Publish the completed full-application mechanism observation once."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-direct-depthwise-observer-amd-20260925'
BEFORE=ROOT/'artifacts/parakeet-depthwise-route-amd-20260925'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def main():
    closed=read(BASE/'closed.json');analysis=read(BASE/'analysis.json');build=read(BASE/'build-review.json')
    before=read(BEFORE/'analysis.json');previous=read(BEFORE/'closed.json')
    assert closed['passed'] and closed['analysis']==pin(BASE/'analysis.json')
    assert previous['passed'] and previous['analysis']==pin(BEFORE/'analysis.json')
    assert analysis['passed'] and analysis['exact_public_results'] and analysis['public_requests']==80
    assert analysis['observed']['zero_generic_work'] and analysis['observed']['every_geometry_exact']
    raw=BASE/'capture-collected/logs/counts.json';assert pin(raw)==analysis['raw_counts']
    oldraw=BEFORE/'capture-collected/logs/counts.json';assert pin(oldraw)==before['raw_counts']
    newrows={r['key']:r for r in read(raw)['rows']};oldrows={r['key']:r for r in read(oldraw)['rows']}
    assert newrows.keys()==oldrows.keys()
    for key,row in newrows.items():
        assert row['layouts']==oldrows[key]['layouts'] and row['calls']==oldrows[key]['calls']
        assert row['direct_batches']==row['completed_batches']==row['calls']
    spec=read(BASE/'bundle/spec.json');assert build['passed'] and build['data_unchanged'] and build['consumer_unchanged']
    report=dict(passed=True,diagnostic_only=True,instrumented_times_not_scored=True,
        closure=pin(BASE/'closed.json'),previous_closure=pin(BEFORE/'closed.json'),
        uninstrumented_product=spec['before_product'],observed_product=analysis['products'],
        before=before['observed']['per_corpus'],after=analysis['observed']['per_corpus'],
        all_geometry_counters=read(raw)['rows'],public_requests=80,exact_public_results=True,
        resources=analysis['resources'],numerical_full_models_pending=True,performance_pending=True,release_admitted=False)
    with (OUT/'mechanism-20260925.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
    lines=['# Direct depthwise: the predicted overhead disappears in the complete application', '',
        'All **80 public transcription requests** match the original M78 results',
        'exactly: transcripts, tokens and decoder calls. Every targeted convolution',
        'takes the direct route across **all 59 observed geometries**. All 2,080',
        'operator calls finish through that route over four corpus passes.', '',
        '| Work per 20-clip corpus | Original M78 | Direct candidate |',
        '| --- | ---: | ---: |']
    for label,key in [('Operator calls','calls'),('Tiled panels','panels'),('Generic matrix products','products'),
                      ('Temporary group tensor views','views'),('Float values written into patches','patch_values')]:
        lines.append(f"| {label} | {report['before'][key]:,} | {report['after'][key]:,} |")
    lines+=['',
        'All 520 eligible calls per corpus use direct accumulation. Source/dense',
        'layouts and frequencies match the original observation at every geometry.',
        'No target call rents tiled scratch or enters a generic matrix leaf.', '',
        'The observer changes only three callsites; all four optimization helpers',
        'remain unchanged. It adds 16 private observer methods, with 3,278 original',
        'Core methods and all 697 Data methods unchanged. Data and the common',
        'public consumer are reused byte-for-byte; there are no added warnings.', '',
        'All 434 resource samples and original CPU-accounting checks pass. Peak',
        'owned RSS is 8,135,192,576 bytes. The actual owner and worker are terminal',
        'with exit code zero. These instrumented times are deliberately unscored.', '',
        'This confirms the proposed mechanism. Full model arrays must still pass',
        'the native ORT comparison, followed by the shape-weighted screen and',
        'complete application/regression gates on the uninstrumented candidate.',
        'The root product and BENCHMARK.md retain their qualified release results.', '',
        f"Uninstrumented Core: `{report['uninstrumented_product']['Lokad.Onnx.dll']['sha256']}`.",
        f"Mechanism closure: `{report['closure']['sha256']}`.", '',
        '[All per-geometry counts](mechanism-20260925.json),',
        '[focused numerical qualification](contracts-20260925.md),',
        '[original ORT-guided diagnosis](../depthwise-route-results/diagnosis-20260925.md).', '']
    with (OUT/'mechanism-20260925.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(report=pin(OUT/'mechanism-20260925.md'),evidence=pin(OUT/'mechanism-20260925.json'))))


if __name__=='__main__':main()
