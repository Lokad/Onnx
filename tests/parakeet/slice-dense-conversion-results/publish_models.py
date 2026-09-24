"""Publish only complete full-model qualification, with no timing promotion."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    outputs=[OUT/'models-20260925.json',OUT/'models-20260925.md'];assert not any(p.exists() for p in outputs)
    closure=read(BASE/'closed.json');analysis=read(BASE/'analysis.json')
    assert closure['passed'] and analysis['passed'] and closure['analysis']==pin(BASE/'analysis.json')
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    assert analysis['reference_provenance_verified'] and analysis['no_performance_measurement']
    results=analysis['results'];assert len(results)==8
    native=[r['native'] for n,r in results.items() if '-native-' in n]
    public=[r for n,r in results.items() if '-public-' in n]
    assert len(native)==len(public)==4
    assert sum(r['arrays'] for r in native)==3136 and sum(r['values'] for r in native)==12361976
    assert all(r['audit_consistent'] and r['application_passed'] and r['numeric_gate_passed'] and not r['failures'] for r in native)
    assert sum(r['public_requests'] for r in public)==80
    for mode in ['512','256']:
        exact=results['candidate-native-'+mode]['native']['exact_selected_comparisons']
        assert len(exact)==784 and all(r['bit_identical'] for r in exact)
        assert results['candidate-public-'+mode]['complete_selected_results_exact']
    maximum=max(r['maximum'] for r in native);assert maximum<=1e-4
    resources=analysis['resources'];samples=sum(r['samples'] for r in resources);peak=max(r['peak_rss'] for r in resources)
    report=dict(passed=True,closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),
        identities=analysis['identities'],consumers=analysis['consumers'],arrays=3136,values=12361976,public_requests=80,
        exact_current_arrays_and_transcripts=True,native_maximum_scaled_error=maximum,native_bound=1e-4,
        resources=resources,resource_samples=samples,peak_rss=peak,performance_measured=False,product_integrated=False,publisher=pin(Path(__file__)))
    with outputs[0].open('x',encoding='utf8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    lines=['# Full Parakeet correctness passes for the slice-copy prototype','',
        'All **3,136 arrays / 12,361,976 values and 80 public transcriptions pass**',
        'across the current release and candidate in both instruction modes.',
        'Candidate arrays and complete public results match the current release',
        'exactly. Native numerical bounds, immutable inputs and held-output ownership',
        'remain unchanged. These are fresh managed executions against retained, pinned',
        'ORT reference arrays; fresh ORT application timing remains a later stage.','',
        f'Maximum native scaled error is **{maximum:.12g}**, below **1e-4**.',
        f'All eight workers and the supervisor are terminal; all **{samples:,}** resource',
        f'samples pass, with peak owned RSS **{peak:,} bytes**. No worker was repeated.',
        'The original 11 GiB preflight and 12 GiB RSS limits stayed in force.','',
        'Both products use ordinary Data `a893952f`; current Core is `f95a13c5` and',
        'candidate Core is `49c3a958`. The qualified native/public consumers were',
        'reused without a build. The candidate adds only the reviewed slice dense',
        'conversion override and already passes 395 tensor cases in each mode.','',
        '**No speedup is claimed or integrated.** Next profile the complete executed',
        'positional group: 34 managed nodes versus 30 optimized ORT nodes, including',
        'shared preparation once and all 24 projections. Six managed constant aliases',
        'are verified. The raw export has 37 nodes before constant deduplication.',
        'Then apply the original complete-application and release gates. BENCHMARK.md',
        'continues to describe the qualified release at 1.632486 times ORT latency.','',
        '[Identities, checks and resource evidence](models-20260925.json),',
        '[exact attribution boundaries](groups-20260925.json),',
        '[frozen full-model protocol](../slice-dense-conversion-models-amd/README.md),',
        '[prototype and retained qualification incidents](results-20260924.md).','']
    with outputs[1].open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(passed=True,report=pin(outputs[0]),native_maximum=maximum,resource_samples=samples,peak_rss=peak)))


if __name__=='__main__':main()
