"""Prepare portable regression tests without copying a historical product DLL."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
SOURCE = ROOT/'artifacts/parakeet-direct-depthwise-source-v2-20260925'
BUILD = ROOT/'artifacts/parakeet-direct-depthwise-build-v2-amd-20260925'
OUT = ROOT/'artifacts/parakeet-direct-depthwise-public-tests-20260925'
TARGET = TOOLS/'DirectDepthwiseTests.cs.txt'
NAME = 'tests/Lokad.Onnx.Backend.Tests/DirectDepthwiseTests.cs'
RETAINED = ['SpatialBordersAndFlattenedVectorTailsMatch', 'LineTailsAndMemoryOffsetsMatch',
            'SpecialValuesIncludingPaddingMatch', 'UnsupportedOptionsAndGeometriesKeepOriginalResults',
            'LogicalViewsAndPublicOutputsAreIndependent', 'ProviderOneDimensionalAndPooledSpatialOutputsMatch']


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def fact(text, name):
    start = text.index('    [Fact]\n    public void '+name+'(')
    next_fact = text.find('    [Fact]\n', start+1)
    end = next_fact if next_fact >= 0 else text.rfind('\n}')
    return text[start:end].rstrip()


def portable(original, geometries):
    text = original
    for name in ['System.Diagnostics', 'System.Reflection', 'System.Runtime.Loader',
                 'System.Security.Cryptography', 'System.Text.Json']:
        text = text.replace('using '+name+';\n', '')
    start = text.index('    static readonly string OriginalDirectory')
    end = text.index('    static float[] Values', start)
    text = text[:start]+text[end:]
    start = text.index('    static object OldTensor(')
    end = text.index('    static void EqualBits(', start)
    text = text[:start]+'''    static DenseTensor<float> ReferenceTensor(Tensor<float> input)
    {
        var value = DenseTensor<float>.OfShape(input.Dimensions.ToArray());
        input.ToDenseTensor().Buffer.Span.CopyTo(value.Buffer.Span);
        return value;
    }

'''+text[end:]
    start = text.index('        object oldOptions =')
    end = text.index('        var scratch =', start)
    text = text[:start]+'''        // The direct path requires one worker. This exercises the existing
        // generic convolution with the same SIMD mode and tensor values.
        var expected = (DenseTensor<float>)Tensor<float>.Conv2D(ReferenceTensor(x), ReferenceTensor(w),
            group, pads, b is null ? null : ReferenceTensor(b), null, strides, dilation,
            options with { MaxDegreeOfParallelism = 2 });
        var reference = expected.Buffer.ToArray();
'''+text[end:]
    before = 'expectDirect && options.UseIntrinsics && Avx2.IsSupported'
    assert text.count(before) == 1
    text = text.replace(before, before+' && Fma.IsSupported')
    text = text.replace(fact(text, 'ActualRuntimeAndProductsMatch')+'\n\n', '')
    before = fact(text, 'EveryObservedGeometryMatchesOriginalBits')
    rows = '\n'.join('            new[] { '+', '.join(map(str, g))+' },' for g in geometries)
    after = '''    [Fact]
    public void EveryObservedGeometryMatchesGenericBits()
    {
        // Recorded Parakeet geometries, independent of model files and VM paths.
        int[][] geometries =
        {
'''+rows+'''
        };
        long values = 0;
        foreach (var g in geometries)
        {
            var x = Data(g[..4]); var w = Data(new[] { g[4], 1, g[5], g[6] }, 1729); var b = Data(new[] { g[4] }, 6131);
            var output = Check(x, w, b, g[17], g[11..15], g[9..11], g[7..9]);
            Assert.Equal(new[] { g[0], g[4], g[15], g[16] }, output.Dimensions.ToArray());
            values += output.Length;
        }
        Assert.Equal(59, geometries.Length);
        Assert.Equal(57332736L, values);
    }'''
    text = text.replace(before, after)
    assert all(fact(original, name) == fact(text, name) for name in RETAINED)
    assert text.count('[Fact]') == 7
    assert all(token not in text for token in ['Environment.', 'File.', 'Assembly', 'OriginalDirectory',
                                               'OriginalDense', 'OriginalOptions', 'OriginalConv', 'OldTensor(', 'Json'])
    return text


def main():
    assert not OUT.exists() and not TARGET.exists()
    assert pin(SOURCE/'prepared.json')['sha256'] == '3b82260e7baca95e13a6e28ea4828455555c26061f4b05764bb0d91e72533b1c'
    assert pin(SOURCE/'source'/NAME) == read(SOURCE/'prepared.json')['source'][NAME]
    assert pin(BUILD/'closed.json')['sha256'] == '26e4da0a66bf37aeda09dd5b0a5144817585ab7311cf87f85c4bb9ded894d171'
    proof = read(BUILD/'closed.json'); assert proof['passed'] and proof['analysis'] == pin(BUILD/'analysis.json')
    analysis = read(BUILD/'analysis.json')
    geometries = [r['geometry'] for r in analysis['suites'][0]['geometry']['records']]
    assert len(geometries) == 59 and all(len(g) == 18 for g in geometries)
    for suite in analysis['suites']:
        assert suite['passed'] == 8 and suite['geometry']['checked_values'] == 57332736
        assert [r['geometry'] for r in suite['geometry']['records']] == geometries
        assert all(r['bitwise_equal'] for r in suite['geometry']['records'])
    original = (SOURCE/'source'/NAME).read_text(encoding='utf8')
    actual = portable(original, geometries)
    OUT.mkdir(); TARGET.write_text(actual, encoding='utf8')
    result = dict(passed=True, prepared_source_only=True, compiled=False, executed=False,
                  root_product_changed=False, release_admitted=False, source=pin(SOURCE/'source'/NAME),
                  qualified_original_tests=pin(BUILD/'closed.json'), evidence=pin(BUILD/'analysis.json'),
                  output=pin(TARGET), preserved_test_bodies=RETAINED, geometries=geometries,
                  expected_public_facts=7, external_runtime_identity_fact_excluded=True,
                  public_oracle='Existing generic convolution with MaxDegreeOfParallelism=2; exact float bits',
                  original_cross_assembly_oracle_preserved_in_closed_evidence=True,
                  preparer=pin(Path(__file__)))
    with (OUT/'prepared.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(prepared=pin(OUT/'prepared.json'), output=pin(TARGET), retained_test_bodies=len(RETAINED),
                          public_facts=7, geometries=len(geometries), compiled=False, executed=False)))


if __name__ == '__main__':
    main()
