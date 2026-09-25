"""Publish the completed representation proof without rerunning its consumer."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-packed-logical-weight-amd-v2-20260925'
OLD=ROOT/'artifacts/parakeet-packed-logical-weight-amd-20260925'


def read(path):return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    target=OUT/'representation-20260925.json';assert not target.exists()
    closure=read(BASE/'closed.json')
    assert pin(BASE/'closed.json')['sha256']=='021b990189a2162d9b0b2db991d789434dba177c333b4fa5b97526661e9f8e89'
    assert closure['passed'] and closure['analysis']==pin(BASE/'analysis.json')
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');build=read(BASE/'build-review.json')
    assert analysis['passed'] and analysis['representation_passed']==closure['representation_passed']
    assert analysis['cases']==48 and not analysis['release_admitted'] and analysis['no_model_execution']
    assert build['passed'] and build['zero_added_warnings'] and not build['product_rebuilt']
    recovery=read(ROOT/'tests/parakeet/packed-logical-weight-probe-v2/recovery.json')
    assert recovery['original_build_output']==pin(OLD/'build-collected/logs/consumer-build.stdout')
    assert recovery['original_collection']==pin(OLD/'build-collected/build-collection.json')
    assert not (OLD/'build-review.json').exists() and not (OLD/'capture-deployment.json').exists()
    for mode,result in analysis['modes'].items():
        assert len(result['cases'])==(29 if mode=='native' else 19)
        assert result['passed']==all(r['passed'] for r in result['cases'])
    result=dict(passed=True,diagnostic_only=True,release_admitted=False,application_saving_measured=False,
        closure=pin(BASE/'closed.json'),build_review=pin(BASE/'build-review.json'),analysis=analysis,
        original_build_rejected_for_warning=recovery,
        representation_source=pin(ROOT/'tests/parakeet/packed-logical-weight-probe-v2/PackedLogicalWeight.cs.txt'),
        consumer_source=pin(ROOT/'tests/parakeet/packed-logical-weight-probe-v2/Program.cs.txt'),
        next_step='Integrate this representation in one isolated private-encoder candidate, preserving existing kernel selection, alias protection, logical fallbacks and default graph behavior.',
        publisher=pin(Path(__file__)))
    with target.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(published=pin(target),representation_passed=analysis['representation_passed'],cases=analysis['cases'])))


if __name__=='__main__':main()
