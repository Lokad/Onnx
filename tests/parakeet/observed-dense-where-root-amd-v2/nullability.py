"""Bind the failed root and the exact nullable-output source correction."""
import hashlib
from pathlib import Path
from protocol import pin,read,JOBS

ROOT=Path(__file__).resolve().parents[3]
INCIDENT=ROOT/'artifacts/parakeet-observed-dense-where-root-amd-20260924'
INITIAL=ROOT/'artifacts/parakeet-observed-dense-where-root-integration-20260924'
HELPER='src/Lokad.Onnx/Zzz.DenseScalarWhere.cs'
FAILURE='8888b3803c6495dedc87111388e021a4132307bae0dbc490efa6333a5a0a4891'
APPLIED='b5aa5ac2b91c4a9617d4db5892a8b88c037e0a24805be8fcffd19631abfe4db8'


def correct(raw):
    replacements=[
        (b'using System.Runtime.CompilerServices;',b'using System.Diagnostics.CodeAnalysis;\r\nusing System.Runtime.CompilerServices;' if b'\r\n' in raw else b'using System.Diagnostics.CodeAnalysis;\nusing System.Runtime.CompilerServices;'),
        (b'Tensor<T> y, out Tensor<T> output)',b'Tensor<T> y, [NotNullWhen(true)] out Tensor<T>? output)'),
        (b'output = null!;',b'output = null;')]
    result=raw
    for before,after in replacements:
        assert result.count(before)==1 and after not in result
        result=result.replace(before,after)
    restored=result
    for before,after in reversed(replacements):
        assert restored.count(after)==1
        restored=restored.replace(after,before)
    assert restored==raw and b'null!' not in result
    return result


def bytes_pin(raw):
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def verify_failure():
    assert pin(INCIDENT/'closed.json')['sha256']==FAILURE
    proof=read(INCIDENT/'closed.json')
    assert not proof['passed'] and not proof['release_admitted']
    assert proof['verifier']==pin(ROOT/'tests/parakeet/observed-dense-where-results/audit_root_failure.py')
    for name,wanted in proof['files'].items():assert pin(INCIDENT/name)==wanted,name
    value=read(INCIDENT/'failure-analysis.json')
    assert proof['analysis']==pin(INCIDENT/'failure-analysis.json')
    assert value['product_tests_failed']==1 and (value['tensor_passed'],value['tensor_failed'])==(367,1)
    assert value['failed_test']=='Lokad.Onnx.Tensors.Tests.NoNullForgivingTests.SourceTree_HasNoNullForgivingOperators'
    assert value['other_tensor_outcomes_exact'] and value['root_source_verified']
    assert value['completed_jobs']==JOBS[:10] and value['unexecuted_jobs']==JOBS[10:]
    assert (value['backend']['passed'],value['backend']['skipped'])==(3499,41)
    assert value['inventory']['implementation_flags_equal']
    assert value['policy_test']==pin(ROOT/'tests/Lokad.Onnx.Tensors.Tests/NoNullForgivingTests.cs')
    assert pin(INITIAL/'applied.json')['sha256']==APPLIED
    assert read(INITIAL/'applied.json')['source_files'][HELPER]==value['helper']
    return value
