"""Integrate the proved final-row helper into one isolated, source-pinned candidate."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-packed-final-row-source-20260925'
ORIGINAL = ROOT/'artifacts/parakeet-owned-packed-weight-scope-source-20260925'
RECOVERY = ROOT/'artifacts/parakeet-owned-packed-weight-scope-recovery-amd-20260925'
PROOF = ROOT/'artifacts/parakeet-packed-final-row-proof-amd-20260925'
TOOLS = Path(__file__).resolve().parent
TEST = 'tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs'
ROUTE = 'src/Lokad.Onnx/TensorOps.OwnedPackedMatMul.cs'
HELPER = 'src/Lokad.Onnx/PackedFinalRowKernel.cs'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def inputs():
    assert pin(ORIGINAL/'prepared.json')['sha256'] == '0d6b78df6e7d03863b15c42af0b5e5bff564ffdf89ef5bc77ef34307f6e1e3a1'
    original = read(ORIGINAL/'prepared.json')
    assert original['passed'] and len(original['source']) == 432
    for name, wanted in original['source'].items(): assert pin(ORIGINAL/'source'/name) == wanted, name
    for folder, digest in [(RECOVERY, '0f8bdc94e2aaf27956af6767093acdb8b57c31655713c9af1a6302d2f7aa31ae'),
                           (PROOF, 'f4cc1ea804d95014ac878757f8f5107b220a940223b1497bbf6ab55136ba415f')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        closure = read(folder/'closed.json')
        assert closure['passed'] and closure['analysis'] == pin(folder/'analysis.json')
        for name, wanted in closure['files'].items(): assert pin(folder/name) == wanted, name
    proof = read(PROOF/'analysis.json')
    assert proof['proof_passed'] and proof['cases'] == 98 and not proof['failures']
    assert proof['no_product_change'] and proof['no_model_execution'] and not proof['prior_quantitative_attribution']
    contracts = read(RECOVERY/'analysis.json')
    assert contracts['passed'] and contracts['product'] == proof['product']
    assert contracts['compiled_review'] == pin(RECOVERY/'build-review.json')
    compiled = read(RECOVERY/'build-review.json'); assert compiled['passed']
    raw_path = RECOVERY/'build-collected/logs/instructions.json'
    assert compiled['inventory'] == pin(raw_path)
    inventory = read(raw_path); assert inventory['inventory_complete']
    before = []
    for row in inventory['observations']:
        methods = {name: body for name, body in row['normalized_methods'].items() if name not in row['removed']}
        methods.update(row['candidate_methods'])
        assert set(methods) == set(row['method_flags_after'])
        assert row['after_sha256'] == proof['product'][row['assembly']]['sha256']
        before.append(dict(assembly=row['assembly'], before_sha256=row['after_sha256'], normalized_methods=methods,
                           method_flags_before=row['method_flags_after'], public_surface=row['public_surface_after']))
    assert [len(row['normalized_methods']) for row in before] == [3277, 697]
    helper = PROOF/'capture-collected/source/PackedFinalRowKernel.cs'
    assert pin(helper) == proof['helper'] == pin(TOOLS.parent/'packed-final-row-probe/PackedFinalRowKernel.cs.txt')
    return original, proof, dict(observations=before), helper


def main():
    assert not BASE.exists()
    original, proof, inventory, helper = inputs()
    old = {name: (ORIGINAL/'source'/name).read_bytes() for name in original['source']}
    values = dict(old); edits = []
    def replace(name, before, after, count=1):
        assert before != after and values[name].count(before.encode()) == count, (name, before)
        values[name] = values[name].replace(before.encode(), after.encode())
        edits.append(dict(file=name, before=before, after=after, count=count))
    corrected = (RECOVERY/'bundle/source'/TEST).read_bytes()
    before = b'graph.Nodes[0].Name = "/pre_encode/out/MatMul";'
    after = b'var node = graph.Nodes[0]; node.Name = "/pre_encode/out/MatMul"; graph.Nodes[0] = node;'
    assert old[TEST].count(before) == 1 and old[TEST].replace(before, after) == corrected
    replace(TEST, before.decode(), after.decode())
    replace(ROUTE, 'using System.Buffers;\n', '')
    removal = '''    static DenseTensor<float>? OwnedRemainderSource(OwnedPackedTensor packed, int m, TensorExecutionOptions options)
    {
        if ((m & 1) == 0 || m % 3 == 0) return null;
        StartOpStage(OpStage.CopyY);
        return CountedCopy(packed.ToDenseTensor(), options.CopyReporter);
    }

'''
    replace(ROUTE, removal, '')
    replace(ROUTE, 'float* packed, float* original, float* output)', 'float* packed, float* output)')
    replace(ROUTE, 'MathOps.ShortWideMultiplyRemainder(1, n, k, x + rows * n, original, output + rows * k);',
            'PackedFinalRowKernel.Multiply(n, k, x + rows * n, packed, output + rows * k);')
    replace(ROUTE, '        var original = OwnedRemainderSource(packed, m, options);\n', '', 2)
    replace(ROUTE, '        using var yp = original is null ? default(MemoryHandle) : original.Buffer.Pin();\n', '', 2)
    replace(ROUTE, '(float*)xp.Pointer, pp, (float*)yp.Pointer, (float*)zp.Pointer)', '(float*)xp.Pointer, pp, (float*)zp.Pointer)')
    replace(ROUTE, 'x + ox, pp, (float*)yp.Pointer, output + oz)', 'x + ox, pp, output + oz)')
    replace(TEST, 'ActualShapesUseExistingArithmetic_WithOnlyRequiredRemainderCopy', 'ActualShapesUseExistingArithmetic_WithoutRemainderReconstruction')
    replace(TEST, 'Assert.Equal(m == 167 ? 16777216L : 0L, context.LastCopyBytes);', 'Assert.Equal(0L, context.LastCopyBytes);')
    for n, k in [(1024, 4096), (4096, 1024)]:
        existing = f'    [InlineData({n}, {k}, 167, true)]'
        extra = '\n'.join(f'    [InlineData({n}, {k}, {m}, {str(m != 89 and m != 50).lower()})]' for m in [89, 157, 83, 61, 151, 169, 50])
        replace(TEST, existing, existing+'\n'+extra)
    assert HELPER not in values; values[HELPER] = helper.read_bytes()
    changed = {name for name in old if values[name] != old[name]}
    assert changed == {ROUTE, TEST} and len(values) == 433
    assert b'CopyY' not in values[ROUTE] and b'OwnedRemainderSource' not in values[ROUTE]
    assert values[ROUTE].count(b'PackedFinalRowKernel.Multiply(') == 1
    for name in old:
        if name not in changed: assert values[name] == old[name]
    BASE.mkdir(); source = BASE/'source'; source.mkdir()
    for name, content in values.items():
        path = source/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(content)
    patch = ''.join(''.join(difflib.unified_diff(old.get(name, b'').decode().splitlines(True), values[name].decode().splitlines(True),
                                                fromfile=name if name in old else '/dev/null', tofile=name)) for name in [ROUTE, HELPER, TEST])
    (BASE/'candidate.patch').write_text(patch, encoding='utf8', newline='\n')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'.agent/m78-parakeet-packed-final-row-20260925.md').read_bytes())
    write(BASE/'prepared.json', dict(passed=True, built=False, root_product_changed=False, release_admitted=False,
        original_source=pin(ORIGINAL/'prepared.json'), corrected_contracts=pin(RECOVERY/'closed.json'),
        corrected_test=pin(RECOVERY/'bundle/source'/TEST), proof=pin(PROOF/'closed.json'), proof_helper=pin(helper),
        proof_consumer=proof['consumer'], proof_runtime='/dev/shm/lokad-parakeet-packed-final-row-proof-20260925/runtime',
        product=proof['product'], inventory=inventory, before=original['source'], source={name: pin(source/name) for name in values},
        edits=edits, added=[HELPER], core_changed_methods=['RunOwnedPackedRows','TryRunOwnedPacked2D','TryRunOwnedPackedBatches'],
        core_removed_methods=['OwnedRemainderSource'], new_helper='Lokad.Onnx.PackedFinalRowKernel::Multiply',
        data_changed_methods=[], public_surface_unchanged=True, existing_arithmetic_methods_unchanged=True,
        expected_owned_count=87, expected_retained_maps=37, expected_reconstructions=0,
        expected_tests={'512':41,'256':41,'scalar':2}, failed_release_controls=proof['failed_release_controls'],
        prior_quantitative_attribution=False, plan=pin(BASE/'prospective-plan.md'), patch=pin(BASE/'candidate.patch'), preparer=pin(Path(__file__))))
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'), source_files=len(values), changed=[ROUTE, HELPER, TEST],
                          expected_tests={'512':41,'256':41,'scalar':2}, root_product_changed=False)))


if __name__ == '__main__': main()
