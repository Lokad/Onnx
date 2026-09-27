"""Adapt the eight arithmetic facts for the portable root suite, preserving the experiment."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'artifacts/parakeet-rational-sigmoid-source-20260927'
BASE = ROOT/'artifacts/parakeet-rational-sigmoid-integration-tests-20260927'
TEST = 'tests/Lokad.Onnx.Backend.Tests/SigmoidVectorTests.cs'
FACTS = ['AllTailsAndZeroLengthMatchScalar', 'SpecialValuesAtEveryVectorLane',
         'ExponentAndRoundingBoundaries', 'DenseFiniteAndBitPatternSweeps',
         'LogicalLayoutsAndOffsetsMatchTheirValues', 'OutputsAreIndependentOfInputsAndLaterCalls',
         'DoublePathRemainsExact', 'InvalidTypesAndOptionsKeepTheirContracts']


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def corrected_fixture(before):
    after = before
    edits = [
        ('    static DenseTensor<float> Run(Tensor<float> input, ExecutionOptions? options = null)',
         '    static DenseTensor<float> Run(Tensor<float> input) => Run(input, null);\n'
         '    static DenseTensor<float> Run(Tensor<float> input, ExecutionOptions? options)'),
        ('    static float Check(float[] values, bool exact = false, ExecutionOptions? options = null)',
         '    static float Check(float[] values) => Check(values, false, null);\n'
         '    static float Check(float[] values, bool exact, ExecutionOptions? options)')]
    for old, new in edits:
        assert after.count(old) == 1
        after = after.replace(old, new)
    marker = '    [Fact]\n    public void ActualRuntimeAndProductsMatchThisQualification()'
    assert after.count(marker) == 1
    offset = after.index(marker)
    removed = after[offset:]
    assert removed.endswith('    }\n}\n') and removed.count('[Fact]') == 1
    after = after[:offset]+'}\n'
    for line in ['using System.Diagnostics;\n', 'using System.Runtime.InteropServices;\n',
                 'using System.Security.Cryptography;\n']:
        assert after.count(line) == 1
        after = after.replace(line, '')
    assert before.count('[Fact]') == 9 and after.count('[Fact]') == 8
    for name in FACTS:
        start = before.index('    [Fact]\n    public void '+name+'()')
        end = before.find('    [Fact]', start+1)
        original = before[start:end].rstrip()
        assert original in after, name
    assert 'options = null' not in after and 'exact = false' not in after
    return after, removed


def main():
    assert not BASE.exists() and not (ROOT/TEST).exists()
    prepared = json.loads((SOURCE/'prepared.json').read_text())
    assert pin(SOURCE/'prepared.json')['sha256'] == '1ca5891c2223e3fd9fa5c65c12c45e3bbeb96f45fc5903d11ccd950895577eb7'
    original = SOURCE/'source'/TEST
    assert pin(original) == prepared['source'][TEST]
    before = original.read_text(encoding='utf8'); after, removed = corrected_fixture(before)
    BASE.mkdir()
    target = BASE/'SigmoidVectorTests.cs'; target.write_text(after, encoding='utf8', newline='\n')
    (BASE/'runtime-guard-retained.txt').write_text(removed, encoding='utf8', newline='\n')
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                    fromfile='experiment/'+TEST, tofile='integration/'+TEST))
    (BASE/'review.patch').write_text(patch, encoding='utf8', newline='\n')
    result = dict(passed=True, source_prepared=pin(SOURCE/'prepared.json'), original=pin(original),
        corrected=pin(target), patch=pin(BASE/'review.patch'), facts=FACTS,
        eight_arithmetic_fact_bodies_unchanged=True, explicit_forwarding_overloads=2,
        qualification_guard_retained=pin(BASE/'runtime-guard-retained.txt'),
        qualification_guard_source=pin(original), product_unchanged=True, root_applied=False,
        source_policy_test=pin(ROOT/'tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'),
        script=pin(Path(__file__)), full_root_suites_still_required=True)
    (BASE/'prepared.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8', newline='\n')
    print(json.dumps(dict(passed=True, fixture=pin(BASE/'prepared.json'), arithmetic_facts=8,
                         explicit_overloads=2, product_unchanged=True, root_applied=False)))


if __name__ == '__main__': main()
