"""Bind the preserved incomplete run and source-proven census correction."""
from pathlib import Path
from protocol import pin,read,JOBS
ROOT=Path(__file__).resolve().parents[3]
INCIDENT=ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-20260923'


def verify_incident():
    assert pin(INCIDENT/'closed.json')['sha256']=='14f5373cead061c8cc5de1847d1cd0fe0b76db6b0dca73137f83b3ca41b744cf'
    proof=read(INCIDENT/'closed.json')
    assert not proof['passed'] and not proof['release_admitted']
    assert pin(ROOT/'tests/parakeet/wide-entry-first-use-results/audit_root_failure.py')==proof['verifier']
    for name,wanted in proof['files'].items():assert pin(INCIDENT/name)==wanted,name
    a=read(INCIDENT/'failure-analysis.json')
    assert not a['passed'] and not a['release_admitted'] and a['product_tests_failed']==0
    assert a['completed_jobs']==JOBS[:11] and a['unexecuted_jobs']==JOBS[11:]
    assert len(a['omitted_cases'])==10 and (a['disabled_backend']['passed'],a['disabled_backend']['skipped'])==(3359,131)
    assert pin(ROOT/'tests/Lokad.Onnx.Backend.Tests/Exp512Tests.cs')==a['source']
    return proof
