"""Publish the closed integration-contract evidence without rerunning any workload."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-tests-amd-20260925'
ORIGINAL=ROOT/'artifacts/parakeet-owned-packed-weight-amd-20260925'


def pin(path):
    return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    closure=read(BASE/'closed.json');analysis=read(BASE/'analysis.json')
    assert pin(BASE/'closed.json')['sha256']=='3fc949cad141d2a1fef1e3913436920f87f3cecb92f1bce94892c1037bc080f4'
    assert closure['passed'] and analysis['passed'] and closure['analysis']==pin(BASE/'analysis.json')
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    compiled=read(ORIGINAL/'build-review.json');tests=read(BASE/'build-review.json')
    assert compiled['passed'] and tests['passed'] and tests['original_compiled_review']==pin(ORIGINAL/'build-review.json')
    assert analysis['original_compiled_review']==pin(ORIGINAL/'build-review.json')
    assert tests['product']==compiled['product']==analysis['product'] and not tests['product_rebuilt']
    failed=read(ORIGINAL/'capture-collected/capture-collection.json')
    assert failed['terminal'] and failed['code']==1
    assert analysis['original_failed_capture']==pin(ORIGINAL/'capture-collected/capture-collection.json')
    assert [(s['mode'],s['passed']) for s in analysis['suites']]==[('512',7),('256',26),('scalar',2)]
    assert len(analysis['retained_native_cases'])==19 and analysis['qualified_native_cases']==26
    source=read(ROOT/'artifacts/parakeet-owned-packed-weight-source-20260925/prepared.json')
    assert len(source['selected'])==87
    allocation_path=ROOT/'artifacts/repository-retention-20260923/m76-contracts-allocation.json'
    allocation=read(allocation_path)
    result=dict(passed=True,release_admitted=False,model_executed=False,application_improvement_measured=False,
        product=analysis['product'],compiled_review=pin(ORIGINAL/'build-review.json'),
        source_prepared=pin(ROOT/'artifacts/parakeet-owned-packed-weight-source-20260925/prepared.json'),
        closure=pin(BASE/'closed.json'),test_review=pin(BASE/'build-review.json'),
        selected_weight_count=87,predicted_owned_payload_bytes=1459617792,
        measured_target_packing_seconds=source['affected_packing_seconds'],
        existing_methods=compiled['methods'],test_method_review=tests['test_methods'],
        suites=analysis['suites'],retained_native_cases=analysis['retained_native_cases'],
        qualified_native_cases=26,qualified_avx512_disabled_cases=26,qualified_hardware_disabled_cases=2,
        resources=analysis['resources'],
        initial_failures=dict(compiled_review_log=pin(ROOT/'artifacts/parakeet-owned-packed-weight-build-review-20260925.log'),
            test_collection=pin(ORIGINAL/'capture-collected/capture-collection.json'),native_passed=24,native_failed=2,
            explanation='Compiler-generated symbol names shifted by five; graph accounting assertions originally read unused caller reporters.'),
        failed_release_controls=analysis['failed_release_controls'],allocation_receipt=pin(allocation_path),
        repository_allocation=allocation,publisher=pin(Path(__file__)))
    destination=TOOLS/'contracts-20260925.json'
    with destination.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(published=pin(destination),product=analysis['product'],qualified_cases=[26,26,2])))


if __name__=='__main__':main()
