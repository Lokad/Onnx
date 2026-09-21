"""Static coverage of the overlapping portable and queued AVX-512 predicates."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PORTABLE = ROOT / 'artifacts/pyannote-portable-integration-tests-20260922'
PAYLOAD = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload'
CENSUS = ROOT / 'artifacts/pyannote-convolution-epilogue-20260921'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def eligibility(rows, reduction, columns):
    # Static shape admission assuming default SIMD/intrinsics, FMA, AVX-512
    # hardware, adequate checked scratch and the generic experiment switch off.
    portable = (rows >= 32 and rows % 2 == 0 and rows % 3 != 0 and reduction >= 64 and columns > 0
        and reduction * columns <= 67108864 and (rows >= 64 or reduction * columns <= 65536))
    avx512 = rows >= 8 and reduction > 0 and columns > 0 and columns % 32 == 0
    return 'both' if portable and avx512 else 'portable_only' if portable else 'avx512_only' if avx512 else 'neither'


def main():
    output = TOOLS / 'dispatch-composition-20260922.json'
    assert not output.exists()
    files = {}
    for path, sha in [(PORTABLE / 'closed.json', 'bf9822426cd75a86cc8c02c065005f0005e1a80b4ee63d5c53313fed145235f6'),
        (CENSUS / 'closed.json', 'aaf2a6457d49f087b536b1966427ce1ab580554a568652f79c23a40b85ea4b84')]:
        assert pin(path)['sha256'] == sha
        proof = read(path)
        assert proof['passed']
        for name, expected in proof['files'].items():
            assert pin(ROOT / name) == expected, name
        files[path.relative_to(ROOT).as_posix()] = pin(path)
    source = PORTABLE / 'source/src/Lokad.Onnx/Zzz.ConvPortableRows.cs'
    native = PAYLOAD / 'source/src/Lokad.Onnx/TensorOps.ConvPackedRows.cs'
    assert pin(native) == read(PAYLOAD / 'payload.json')['files'][native.relative_to(PAYLOAD).as_posix()]
    portable_text, native_text = source.read_text(), native.read_text()
    assert '!AblationSwitches.EnablePackedAvx512Dynamic' in portable_text
    assert 'Avx512F.IsSupported' not in portable_text and 'Avx512F.IsSupported' in native_text
    assert 'columns % 32 == 0' in native_text
    observations = ROOT / 'tests/pyannote/convolution-epilogue/observations-20260921.json'
    census = read(observations)
    assert census['passed'] and not census['inference_executed']
    shapes = [dict(rows=row['m'], reduction=row['n'], columns=row['k'], admission=eligibility(row['m'], row['n'], row['k']))
        for row in census['all_tile_shapes']]
    assert len(shapes) == 22
    counts = collections.Counter(row['admission'] for row in shapes)
    assert counts == dict(both=10, portable_only=8, avx512_only=3, neither=1)
    cases = []
    for case in census['cases']:
        counts_by_call, terms = collections.Counter(), collections.Counter()
        for row in case['rows']:
            assert row['tiled'] and not row['pointwise']
            n, _, height, width = row['output_shape']
            m, c, kh, kw = row['weight_shape']
            groups = row['attributes']['group']
            assert m % groups == 0
            rows, reduction = m // groups, c * kh * kw
            for start in range(0, height * width, row['block_columns']):
                columns = min(row['block_columns'], height * width - start)
                assert columns in row['tile_widths']
                admission = eligibility(rows, reduction, columns)
                counts_by_call[admission] += n * groups
                terms[admission] += n * groups * rows * reduction * columns
        cases.append(dict(name=case['name'], tile_products=dict(counts_by_call), scalar_product_terms=dict(terms),
            overlap_term_share=terms['both'] / sum(terms.values())))
    for path in [source, native, PAYLOAD / 'payload.json', observations, Path(__file__)]:
        files[path.relative_to(ROOT).as_posix()] = pin(path)
    result = dict(passed=True, inference_executed=False, files=files, shapes=shapes, shape_counts=dict(counts), cases=cases,
        scope='Static eligibility under explicit hypothetical hardware/options; no emitted-code, actual runtime dispatch or performance claim. No composition selected.')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(output=str(output.relative_to(ROOT)), shape_counts=dict(counts), cases=cases)))


if __name__ == '__main__':
    main()
