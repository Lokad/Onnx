"""Extract portable regression facts from the closed contracts; do not apply root source."""
import difflib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
FIRST = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-amd-20260927'
CONTRACTS = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-v3-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-rational-sigmoid-root-amd-20260927'
OUT = ROOT/'artifacts/parakeet-decoder-packed-row-integration-tests-20260927'
FACTS = ['DimensionsRowsAndBatchesRetainBitsAndOwnership',
         'OptionsMissingAndReplacedWeightsRetainBitsAndOwnership',
         'RawBoundariesRetainArithmeticGuardsAndZeroAllocation',
         'RawExceptionalValuesRetainNanPayloadsAndOwnedInputs']


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def render(source):
    # Preserve arithmetic, guards, ownership and case values; remove campaign IO,
    # product/CPU identity gates and the externally stored real model fixture.
    assert source.count('    static void Raw(') == source.count('    static void Public(') == 1
    common = source[source.index('    delegate void PackedKernel'):source.index('    static void Raw(')]
    common = common.replace('    static readonly List<object> RawResults = new(), PublicResults = new();\n', '')
    line, = [s for s in common.splitlines(True) if s.startswith('    static string FileHash(')]
    common = common.replace(line, '')
    start, end = common.index('    static void Save('), common.index('    static float[] Pack(')
    common = common[:start]+common[end:]
    raw = source[source.index('    static void Raw('):source.index('    static void Public(')]
    raw = raw.replace('Raw(string folder, string name,', 'Raw(string name,')
    raw = raw[:raw.index('        var output = expected.AsSpan')]+'    }\n'
    public = source[source.index('    static void Public('):source.index('    static void Main(')]
    public = public.replace('Public(string folder, string name,', 'Public(string name,').replace(
        'TensorExecutionOptions mode, string? expectedHash)', 'TensorExecutionOptions mode)')
    for line in [
        '        if (expectedHash is not null) Require(Hash(held) == expectedHash, "Captured projection output");\n',
        '        long allocationStart = GC.GetAllocatedBytesForCurrentThread();\n',
        '        long allocated = GC.GetAllocatedBytesForCurrentThread() - allocationStart;\n']:
        assert public.count(line) == 1; public = public.replace(line, '')
    public = public[:public.index('        Save(folder, "public-"')]+(
        '        Require(copies == 0 && scratches == 0, name + " no input copy or scratch");\n    }\n')
    first = source[source.index('        foreach (int k in new[] { 8191'):source.index('        foreach (var selected in new[]')]
    first = first.replace('Public(output, ', 'Public(').replace(', normal, null)', ', normal)')
    second = source[source.index('        foreach (var selected in new[]'):source.index('        string model = fixture')]
    second = second.replace('Public(output, ', 'Public(').replace(', selected, null)', ', selected)')
    second = second.replace('            int id = PublicResults.Count;', '            int id = index++;')
    third = source[source.index('            foreach (int n in new[] { 0, 1, 3, 17 }'):source.index('            float[] special =')]
    fourth = source[source.index('            float[] special ='):source.index('            Raw(output, "captured",')]
    def unindent(value): return '\n'.join(s[4:] if s.startswith('    ') else s for s in value.splitlines())+'\n'
    third, fourth = [unindent(s.replace('Raw(output, ', 'Raw(')) for s in [third, fourth]]
    header = '''using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;

namespace Lokad.Onnx.Backend.Tests;

// Portable regression coverage extracted from the closed prepared-row contracts.
// The real Parakeet fixture and runtime/product guards remain in that campaign.
public unsafe class PreparedSingleRowTests
{
'''
    methods = []
    for index, body in enumerate([first, second, third, fourth]):
        attribute = 'Fact' if index < 2 else 'SkippableFact'
        prelude = '        var normal = TensorExecutionOptions.Auto;\n' if index < 2 else (
            '        Skip.If(!Fma.IsSupported, "FMA required");\n        PackedKernel kernel = PreparedSingleRowKernel.Multiply;\n')
        if index == 1: prelude += '        int index = 0;\n'
        methods.append(f'\n    [{attribute}]\n    public void {FACTS[index]}()\n    {{\n'+prelude+body+'    }\n')
    value = header+common+raw+public+''.join(methods)+'}\n'
    assert not any(s in value for s in ['PublicResults','RawResults','expectedHash','FileStream','JsonDocument','ProcessorAffinity'])
    assert value.count('[Fact]') == value.count('[SkippableFact]') == 2
    return value


def review():
    inputs = {}
    for folder, filename, digest in [
        (FIRST, 'failed.json', '6225233c00528cde3170a535ac973ddc7382c3af5086b370d90ff860d023d24d'),
        (CONTRACTS, 'closed.json', 'fc00688c8ef0e65e5d5109f1808209add023e740aabc5b4312647e547df7d61d'),
        (QUALIFIED, 'closed.json', 'f6df53ce2ab773898bc84f144abeda97908d383fef9a4df4b7299a22a4d3594d')]:
        assert pin(folder/filename)['sha256'] == digest
        proof = read(folder/filename)
        assert proof.get('passed') or (proof['evidence_verified'] and proof['terminal'] and proof['builds_passed'])
        inputs[(folder/filename).relative_to(ROOT).as_posix()] = pin(folder/filename)
    original = FIRST/'frozen-tools/Contracts.cs.txt'
    assert pin(original) == read(FIRST/'failed.json')['files']['frozen-tools/Contracts.cs.txt']
    inputs[original.relative_to(ROOT).as_posix()] = pin(original)
    source = original.read_text(encoding='utf8'); expected = render(source)
    fixture = TOOLS/'PreparedSingleRowTests.cs.txt'
    assert fixture.read_text(encoding='utf8') == expected
    values = read(CONTRACTS/'analysis.json'); assert values['passed']
    assert read(CONTRACTS/'closed.json')['analysis'] == pin(CONTRACTS/'analysis.json')
    # This additional portable assertion was already true for every closed call.
    for modes in values['contracts'].values():
        for result in modes.values():
            assert all(r['copies'] == r['scratches'] == 0 for r in result['public_cases'])
    policy = 'tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'
    before = read(QUALIFIED/'bundle/evidence/root-applied.json')['source_files']
    assert pin(ROOT/policy) == before[policy]
    stage = FIRST/'bundle/stage.json'
    assert pin(stage) == read(FIRST/'failed.json')['files']['bundle/stage.json']
    assert len(read(stage)['source']) == 440
    for path in [stage, CONTRACTS/'analysis.json', fixture, Path(__file__), ROOT/policy]:
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    return dict(passed=True, product_unchanged=True, root_applied=False, compiled_or_executed=False,
                source_stage=pin(stage), original=pin(original), fixture=pin(fixture), script=pin(Path(__file__)),
                facts=FACTS, public_synthetic_cases=44, raw_synthetic_cases=80,
                original_model_fixture_and_runtime_guards_retained=True, source_policy_test=before[policy], inputs=inputs)


def main():
    assert sys.argv[1:] in [[], ['--prepare']]
    value = review()
    if sys.argv[1:]:
        assert not OUT.exists(); OUT.mkdir()
        source = (FIRST/'frozen-tools/Contracts.cs.txt').read_text(encoding='utf8')
        fixture = (TOOLS/'PreparedSingleRowTests.cs.txt').read_text(encoding='utf8')
        (OUT/'PreparedSingleRowTests.cs').write_text(fixture, encoding='utf8', newline='\n')
        patch = ''.join(difflib.unified_diff(source.splitlines(True), fixture.splitlines(True),
                                            fromfile='closed/Contracts.cs', tofile='PreparedSingleRowTests.cs'))
        (OUT/'review.patch').write_text(patch, encoding='utf8', newline='\n')
        with (OUT/'prepared.json').open('x', encoding='utf8', newline='\n') as stream:
            json.dump(dict(**value, patch=pin(OUT/'review.patch')), stream, indent=2); stream.write('\n')
    else:
        proof = read(OUT/'prepared.json')
        assert proof == dict(**value, patch=pin(OUT/'review.patch'))
        assert pin(OUT/'PreparedSingleRowTests.cs') == value['fixture']
    print(json.dumps({k: value[k] for k in ['passed','facts','public_synthetic_cases','raw_synthetic_cases','root_applied','compiled_or_executed']}))


if __name__ == '__main__': main()
