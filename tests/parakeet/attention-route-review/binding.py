"""Revalidate retained attention route decisions without importing old clocks."""
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
sys.path.insert(1, str(TOOLS.parent / 'pointwise-tail-profile-results'))
from partition import pin, read
from run import native

OLD = ROOT / 'artifacts/parakeet-projection-route-amd-20260924'
ROUTES = ROOT / 'artifacts/parakeet-projection-route-resume-amd-20260924'
SOURCE = ROOT / 'artifacts/parakeet-observed-dense-where-source-20260924'
CENSUS = ROOT / 'artifacts/parakeet-owned-packed-weight-scope-census-amd-20260925'
OWNED_SOURCE = ROOT / 'artifacts/parakeet-owned-packed-weight-scope-source-20260925/source'
MANAGED = ROOT / 'artifacts/parakeet-pointwise-tail-profile-amd-20260927'
NATIVE = native.BASE
GAP = ROOT / 'artifacts/parakeet-pointwise-tail-gap-20260927'


def bind():
    context = native.qualification()
    inputs = {}

    def keep(path, wanted=None):
        actual = pin(path)
        if wanted is not None:
            assert actual == wanted, path
        inputs[path.relative_to(ROOT).as_posix()] = actual
        return read(path)

    proof = keep(ROUTES / 'closed.json')
    assert inputs[(ROUTES / 'closed.json').relative_to(ROOT).as_posix()]['sha256'] == '916d4c688b19688e837a5bd5891efb31bf34a06a925e2431bdd1883b4e7e900d'
    assert proof['passed']
    old_analysis = keep(ROUTES / 'analysis.json', proof['analysis'])
    assert old_analysis['exact_public_results'] and old_analysis['observations'] == 17360
    records = keep(ROUTES / 'observations.json', proof['observations'])
    build = keep(OLD / 'build-review.json', proof['build_review'])
    assert build['passed'] and build['core_unchanged']
    old_spec = keep(OLD / 'bundle/spec.json', build['spec'])
    source = keep(SOURCE / 'prepared.json', old_spec['source_receipt'])
    assert source['passed']
    graph_path = OLD / 'bundle/evidence/graphs.json'
    graph = keep(graph_path, old_spec['files']['evidence/graphs.json'])['encoder-model.onnx']

    closure = keep(MANAGED / 'closed.json')
    assert closure['passed']
    for name, wanted in closure['files'].items():
        assert pin(MANAGED / name) == wanted, name
    current = keep(MANAGED / 'analysis.json', closure['analysis'])
    assert current['complete_admitted_public_results_exact']
    current_spec = keep(MANAGED / 'bundle/spec.json')
    assert current_spec['qualification_closures']['root'] == context['root']
    assert current_spec['reference_product'] == context['measured']
    assert graph == keep(MANAGED / 'capture-collected/wall/graphs.json')['encoder-model.onnx']
    assert graph['retained_packed_bytes'] == graph['maximum_packed_bytes'] == 256 * 1024**2
    assert not any(n['op'] == 'LSTM' for n in graph['nodes'])
    for name in ['encoder-model.onnx', 'encoder-model.onnx.data']:
        remote = '/home/vermorel/Onnx/models/parakeet-tdt-0.6b-v3/' + name
        assert old_spec['external'][remote] == current_spec['external'][remote]

    # Resolve both source endpoints through their measured/qualified receipts.
    applied = keep(native.QUALIFIED_ROOT / 'bundle/evidence/root-applied.json')
    transformations = {}

    def texts(name):
        relative = 'src/Lokad.Onnx/' + name
        old_path, new_path = SOURCE / 'source' / relative, ROOT / relative
        assert pin(old_path) == source['source'][relative]
        assert pin(new_path) == applied['source_files'][relative]
        inputs[old_path.relative_to(ROOT).as_posix()] = pin(old_path)
        inputs[relative] = pin(new_path)
        return old_path.read_text(encoding='utf8'), new_path.read_text(encoding='utf8')

    def restore(name, changes):
        before, after = texts(name)
        for old, new in changes:
            assert after.count(new) == 1, (name, new)
            after = after.replace(new, old, 1)
        assert after == before, name
        transformations[name] = len(changes)

    for name in ['GraphPacking.cs', 'GraphConvPacking.cs', 'Zzz.IsolatedShortMatMul.cs',
                 'MathOps.PackedAvx512.cs', 'AblationSwitches.cs']:
        restore(name, [])
    restore('TensorOps.MatMul.cs', [
        ('        if ((m & 1) != 0 && (m % 3) != 0) return null;',
         '        if (m == 1)\n        {\n            // Consume an existing preparation only in the current wide one-row territory.\n            if (y.Dimensions[^1] < M1BlockedMinColumns) return null;\n        }\n        else if ((m & 1) != 0 && (m % 3) != 0) return null;'),
        ('', '        if (m == 1)\n        {\n            PreparedSingleRowKernel.Multiply(n, k, x, packed, dest);\n            return;\n        }\n'),
        ('        RunBatchedFloatMatMul(bx, by, target, options);',
         '        if (!TryRunOwnedPackedBatches(bx, by, target, options))\n            RunBatchedFloatMatMul(bx, by, target, options);'),
        ('            RunBatchedFloatMatMul(bx, by, z, options);',
         '            if (!TryRunOwnedPackedBatches(bx, by, z, options))\n                RunBatchedFloatMatMul(bx, by, z, options);')])
    restore('Zzz.WideProjectionEntry.cs', [('', '\n        if (TryRunOwnedPacked2D(x, y, destination, options)) return destination;\n')])
    restore('CPUExecutionProvider.MatMul.cs', [('', '        if (t is Tensor<float> owned && OwnedPackedTensor.Resolve(owned) is not null) return t;\n')])
    restore('MathOps.cs', [
        ('', '            int sharedRows = M >= 64 && N >= 64 ? M - M % 8 : 0;\n'),
        ('', '                if (sharedRows > 0)\n                    PackedColumnTailEightRows(sharedRows, N, K, rem, A,\n                        T + tt * Vector256<float>.Count, C + blocked + tt * Vector256<float>.Count);\n'),
        ('for (int i = 0; i < M; i += 2)', 'for (int i = sharedRows; i < M; i += 2)'),
        ('', '            int maskedRows = tail > 0 && Avx2.IsSupported ? sharedRows : 0;\n            if (maskedRows > 0)\n                PackedColumnMaskedEightRows(maskedRows, N, K, rem, tail, A,\n                    T + vcols, C + blocked + vcols);\n'),
        ('for (int i = 0; i < M; i += 2)', 'for (int i = maskedRows; i < M; i += 2)')])
    restore('ComputationalGraph.cs', [('', line + '\n') for line in [
        '        if (typed is OwnedPackedTensor packed) { root = packed.PackedArray; return true; }',
        '        if (typed is OwnedPackedTensor packed) { roots.Add(packed.PackedArray); return true; }',
        '        if (typed is OwnedPackedTensor packed) return ReferenceEquals(candidate, packed.PackedArray);']])
    # Encoder preparation has no LSTM nodes; its prune/admission code is unchanged.
    before, after = texts('GraphLstmPacking.cs')
    marker = '    static PackedLstmWeight Prepare('
    assert before.count(marker) == after.count(marker) == 1
    before = before[before.index('internal sealed record'):before.index(marker)]
    after = after[after.index('internal sealed record'):after.index(marker)]
    assert before == after
    transformations['GraphLstmPacking.cs'] = 'Unchanged preparation before Pack; encoder contains no LSTM'

    # The later owned-weight census observes the unchanged 37 mappings after conversion.
    census_closure = keep(CENSUS / 'closed.json')
    assert pin(CENSUS / 'closed.json')['sha256'] == '577a07a60d901cf57c4ba46d70b2c6fc4c02148ab67a1c28fd78d9b4fbcc3d08'
    assert census_closure['passed']
    compiled_name = 'capture-collected/evidence/compiled-review.json'
    compiled = keep(CENSUS / compiled_name, census_closure['files'][compiled_name])
    assert compiled['passed'] and compiled['existing_arithmetic_unchanged']
    owned_source = keep(OWNED_SOURCE.parent / 'prepared.json', compiled['source'])
    assert owned_source['passed']
    maps = None
    for mode in ['512', '256']:
        name = f'capture-collected/probe/{mode}/after.json'
        after_census = keep(CENSUS / name, census_closure['files'][name])
        name = f'capture-collected/probe/{mode}/before.json'
        before_census = keep(CENSUS / name, census_closure['files'][name])
        name = f'capture-collected/probe/{mode}/result.json'
        result = keep(CENSUS / name, census_closure['files'][name])
        assert result['passed'] and result['core_sha256'] == compiled['product']['Lokad.Onnx.dll']['sha256']
        assert after_census['maps'] == before_census['maps']
        if maps is None:
            maps = after_census['maps']
        else:
            assert maps == after_census['maps']
    assert len(maps) == 37 and sum(m['Bytes'] for m in maps) == 256 * 1024**2
    for relative in ['src/Lokad.Onnx/GraphOwnedPacking.cs', 'src/Lokad.Onnx.Data/ParakeetTranscriber.cs']:
        a, b = OWNED_SOURCE / relative, ROOT / relative
        inputs[a.relative_to(ROOT).as_posix()], inputs[relative] = pin(a), pin(b)
        assert pin(a) == owned_source['source'][relative]
        assert pin(b) == applied['source_files'][relative]
        old, new = a.read_text(encoding='utf8'), b.read_text(encoding='utf8')
        if relative.endswith('GraphOwnedPacking.cs'):
            marker = 'int PrepareOwnedMatMulWeights()'
            assert old[old.index(marker):] == new[new.index(marker):]
            assert 'if (!((n == 1024 && k == 4096) || (n == 4096 && k == 1024))) continue;' in new
            assert 'protectedRoots.Contains(window.Array)' in new
        else:
            assert old == new
            original = SOURCE / 'source' / relative
            assert pin(original) == source['source'][relative]
            inputs[original.relative_to(ROOT).as_posix()] = pin(original)
            addition = '        encoder.PrepareOwnedMatMulWeights();\n'
            assert new.count(addition) == 1 and new.replace(addition, '') == original.read_text(encoding='utf8')
    # These source checks establish route eligibility only. New copy costs and
    # internal arithmetic timings are deliberately not inherited from the old capture.
    return dict(context=context, inputs=inputs, graph=graph, records=records, maps=maps,
                transformations=transformations, current=current,
                interpretation='Retained route metadata revalidated against current source, graph, shapes and preparation; no new internal-route observation')
