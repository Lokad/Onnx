"""Publish complete M76 model correctness without treating worker durations as scores."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'owned-packed-weight-models-amd'))
from run import BASE,prepared,pin,read


def main():
    assert not (HERE/'models-20260925.json').exists() and not (HERE/'models-20260925.md').exists()
    prepared();closure=read(BASE/'closed.json')
    assert pin(BASE/'closed.json')['sha256']=='bb1a758c6936fa251d921992c67dd7c65ff075fe20829af72f4455b529f1b2ca'
    assert closure['passed'] and closure['analysis']==pin(BASE/'analysis.json')
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');payload=read(BASE/'payload.json');rows=[]
    assert analysis['passed'] and analysis['no_performance_measurement']
    for name,result in analysis['results'].items():
        assert result['passed'] and result['no_performance_measurement']
        row=dict(name=name,passed=True)
        if '-native-' in name:
            native=result['native'];assert native['arrays']==784 and native['values']==3090494 and not native['failures']
            row.update(arrays=native['arrays'],values=native['values'],maximum_scaled_error=native['maximum'])
            if name.startswith('candidate-'):
                comparisons=native['exact_selected_comparisons']
                assert len(comparisons)==784 and all(c['bit_identical'] for c in comparisons)
                row['exact_selected_arrays']=len(comparisons)
        else:
            assert result['public_requests']==20
            row['public_requests']=20
            if name.startswith('candidate-'):assert result['complete_selected_results_exact'];row['exact_selected_results']=20
        rows.append(row)
    assert sum(r.get('arrays',0) for r in rows)==3136 and sum(r.get('values',0) for r in rows)==12361976
    assert sum(r.get('public_requests',0) for r in rows)==80
    report=dict(passed=True,performance_measured=False,release_admitted=False,closure=pin(BASE/'closed.json'),
        analysis=pin(BASE/'analysis.json'),identities=analysis['identities'],consumers=analysis['consumers'],jobs=rows,
        resources=analysis['resources'],failed_release_controls=payload['failed_release_controls'],
        pending=['actual reconstruction counters','matched application comparison','unresolved e5 repeatability','remaining release qualification'],
        publisher=pin(Path(__file__)))
    (HERE/'models-20260925.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    prose='''# Complete Parakeet correctness for owned packed weights

The corrected isolated M76 candidate passes all eight full-model jobs. Both
normal and AVX512-disabled execution preserve exact selected-M73 tensor bits,
decoder decisions and complete public transcriptions. The native ORT comparison
passes its unchanged1e-4 scaled-error bound; the maximum remains
3.830720152728115e-5 for both products and modes.

The campaign checks3,136 arrays and12,361,976 values against pinned native
references, plus80 public requests:20 complete corpus clips for each product and
instruction mode. Candidate comparisons require byte-identical saved arrays and
exact public results. Immutable inputs, independently held outputs, actual
runtime identities, instruction environment, CPU affinity and all resource
limits pass. The native consumer constructs ParakeetTranscriber and therefore
exercises the candidate's changed Data constructor and owned weight preparation.

Only product bindings, prerequisites, namespace and provenance labels change in
the retained model protocol. Consumers, numerical/public auditors, worker and
resource limits are unchanged. All PID/birth owners are terminal, and results
are collected and independently audited at closurebb1a758c.

This establishes correctness, not a performance gain. Worker durations include
validation work and must not be used as application scores. Peak candidate RSS
is9.45GB in the normal public worker and7.99GB with AVX512 disabled, below the
12GiB guard; no general process-memory reduction is established by these runs.

Next, count real packing scratch and odd-row reconstructions over the20-clip
encoder corpus, then use the matched complete-application protocol. The predicted
traffic changes are1740 avoided16MiB packs and609 added16MiB reconstructions;
these remain source predictions until the counter check completes. There is no
new kernel or parameter search. Prior e5 repeatability failures still prevent
release promotion, so BENCHMARK.md remains on the qualified release.

Selected: Core49c3a958/Dataa893952f (isolated M73).
Candidate: Core82c02785/Data3f80f8cb (corrected M76).
Full hashes, per-job checks and retained release failures are in
[models-20260925.json](models-20260925.json). Raw artifacts are retained in
artifacts/parakeet-owned-packed-weight-models-amd-20260925.
'''
    (HERE/'models-20260925.md').write_text(prose,encoding='utf8')
    print(json.dumps(dict(report=pin(HERE/'models-20260925.json'),closure=pin(BASE/'closed.json'),arrays=3136,values=12361976,public_requests=80)))


if __name__=='__main__':main()
