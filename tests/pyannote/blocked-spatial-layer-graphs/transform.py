"""Adapt the qualified fixture checker only at graph dispatch and identity boundaries."""
import difflib


def transform(source, core):
    old = '''        internal DenseTensor<float> Candidate(int lanes)
        {
            if (!Eligible) return Production();
            return new DenseTensor<float>(BlockedSpatial.Run(Input, Prepared, Bias, Residual,
                InputShape[1], OutputShape[1], InputShape[2], InputShape[3], Strides[0], lanes, Relu), OutputShape);
        }'''
    assert source.count(old) == 1
    result = source.replace(old, '        internal DenseTensor<float> Candidate(int lanes) => GraphCalls.Run(this, lanes);')
    before = '1279b4b662241db2404fa4875eae20eaa924f15677a85655f99b8f81cd24b309'
    assert result.count(before) == 1; result = result.replace(before, core)
    # AVX2 qualification on an AVX512 host may explicitly disable only AVX512F.
    old = '        Require(flags.Length == 0, "Unexpected runtime flags");'
    new = '''        Require(flags.Length == 0 || (args[0] == "256" && flags.SequenceEqual(new[] { "DOTNET_EnableAVX512F" })
            && Environment.GetEnvironmentVariable("DOTNET_EnableAVX512F") == "0"), "Unexpected runtime flags");'''
    assert result.count(old) == 1; result = result.replace(old, new)
    old = '        fixtures.Unchanged();'; assert result.count(old) == 1
    result = result.replace(old, old+' GraphCalls.Complete();')
    old = '            pid = Environment.ProcessId, flags, no_performance_measurement = true };'
    new = '''            pid = Environment.ProcessId, flags, no_performance_measurement = true,
            layer_graphs = GraphCalls.Count, graph_retained_bytes = GraphCalls.RetainedBytes, graph_dispatch = GraphCalls.Rows };'''
    assert result.count(old) == 1; result = result.replace(old, new)
    return result, ''.join(difflib.unified_diff(source.splitlines(True), result.splitlines(True), fromfile='qualified-component/ModelProbe.cs', tofile='normal-layer-graphs/ModelProbe.cs'))
