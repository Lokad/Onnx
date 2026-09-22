"""Anchored normal-product composition; no mutation of selected root source."""
import difflib
from pathlib import Path


def component_files(component):
    result = {}
    for name, target in [('BlockedSpatial.cs', 'Zzz.ConvBlockedSpatial.cs'),
                         ('GeneratedKernels.cs', 'Zzz.ConvBlockedSpatial.Kernels.cs'),
                         ('VectorInput.cs', 'Zzz.ConvBlockedSpatial.Input.cs'),
                         ('VectorEpilogue.cs', 'Zzz.ConvBlockedSpatial.Output.cs')]:
        value = (component/name).read_text()
        if name == 'BlockedSpatial.cs':
            start = value.index('        if (!Finite(input)'); end = value.index('        PackInput(input,', start)
            value = value[:start] + '        if (!Finite(input) || !Finite(prepared) || !Finite(bias) || !Finite(residual)) return false;\n' + value[end:]
            start = value.index('    internal static float[] Run(')
            value = value[:start] + '}\n'
        value = value.replace('internal static unsafe partial class BlockedSpatial', 'namespace Lokad.Onnx;\n\ninternal static unsafe partial class ConvBlockedSpatial')
        assert 'class BlockedSpatial' not in value
        result['src/Lokad.Onnx/'+target] = value
    return result


def changes(source, component, tools):
    result = component_files(component)
    for name in ['GraphConvPacking', 'TensorOps.ConvBlocked']:
        result['src/Lokad.Onnx/'+name+'.cs'] = (tools/(name+'.cs.txt')).read_text()
    edits = {
        'src/Lokad.Onnx/ComputationalGraph.cs': [
            ('    internal Dictionary<float[], PackedMatMulWeight> PackedWeights = new Dictionary<float[], PackedMatMulWeight>();',
             '    internal Dictionary<float[], PackedMatMulWeight> PackedWeights = new Dictionary<float[], PackedMatMulWeight>();\n    internal Dictionary<float[], PackedConvWeight> PackedConvWeights = new Dictionary<float[], PackedConvWeight>();'),
            ('                PackedWeights.Clear();', '                PackedWeights.Clear();\n                PackedConvWeights.Clear();'),
            ('PackedMatMulWeights = PackedWeights }', 'PackedMatMulWeights = PackedWeights, PackedConvWeights = PackedConvWeights }'),
            ('        GraphPacking.PackMatMulWeights(this);', '        GraphPacking.PackMatMulWeights(this);\n        GraphConvPacking.PackWeights(this);'),
            ('Retained panel-packed MatMul weight bytes held by the prepared plan.', 'Retained prepared matrix and convolution weight bytes held by the prepared plan.'),
            ('Set by preparation (PackMatMulWeights), not per execution', 'Set by shared matrix/convolution weight preparation, not per execution')],
        'src/Lokad.Onnx/GraphPacking.cs': [('        long retained = 0;', '        long retained = GraphConvPacking.PruneAndBytes(graph);')],
        'src/Lokad.Onnx/GraphExecution.cs': [('        PackedWeights = prepared.PackedWeights;', '        PackedWeights = prepared.PackedWeights;\n        PackedConvWeights = prepared.PackedConvWeights;')],
        'src/Lokad.Onnx/TensorExecutionOptions.cs': [('    internal IReadOnlyDictionary<float[], PackedMatMulWeight>? PackedMatMulWeights { get; init; }',
            '    internal IReadOnlyDictionary<float[], PackedMatMulWeight>? PackedMatMulWeights { get; init; }\n\n    /// <summary>Optional immutable prepared convolution weights shared by graph contexts.</summary>\n    internal IReadOnlyDictionary<float[], PackedConvWeight>? PackedConvWeights { get; init; }')],
        'src/Lokad.Onnx/TensorOps.ConvPool.cs': [('        bool hasBias = bd is not null;\n        if (kH == 1',
            '        bool hasBias = bd is not null;\n        if (TryConvBlockedSpatial(xd, weight, wd, bd, output, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, options)) return output;\n        if (kH == 1')]
    }
    for name, replacements in edits.items():
        value = (source/name).read_text(encoding='utf-8-sig')
        for before, after in replacements:
            assert value.count(before) == 1, (name, before, value.count(before)); value = value.replace(before, after)
        result[name] = value
    for name in ['ConvBlockedSpatialTests.cs']:
        result['tests/Lokad.Onnx.Backend.Tests/'+name] = (tools/(name+'.txt')).read_text()
    return result


def apply(source, component, tools):
    patch = ''
    for name, after in changes(source, component, tools).items():
        path = source/name; before = path.read_text(encoding='utf-8-sig') if path.exists() else ''
        patch += ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/'+name if path.exists() else '/dev/null', tofile='b/'+name))
        path.write_text(after, encoding='utf8')
    return patch
