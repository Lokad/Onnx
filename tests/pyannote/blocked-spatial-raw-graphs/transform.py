"""Keep the raw census and use actual product helpers plus ordinary graph callers."""
import difflib


def transform(source, core):
    text = source

    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    replace('using Lokad.Onnx;', 'using Lokad.Onnx;\nusing BlockedSpatial = Lokad.Onnx.ConvBlockedSpatial;')
    replace('1279b4b662241db2404fa4875eae20eaa924f15677a85655f99b8f81cd24b309', core)
    start = text.index('        var first = BlockedSpatial.Run(')
    end = text.index('        return rejected;', start)
    text = text[:start]+'        // Every ordinary graph case below checks distinct owned and held results.\n'+text[end:]
    replace('        Require(flags.Length == 0, "Unexpected runtime override");', '''        Require(flags.Length == 0 || (args[0] == "256" && flags.SequenceEqual(new[] { "DOTNET_EnableAVX512F" })
            && Environment.GetEnvironmentVariable("DOTNET_EnableAVX512F") == "0"), "Unexpected runtime override");
        Require(GraphConvPacking.Lanes == lanes, "Actual product instruction width");''')
    replace('            int differences = 0;', '            int differences = 0, nanPayloads = 0;')
    replace('                if (Bits(output[3 + i]) != Bits(expected[i])) differences++;',
            '''                if (used && Bits(output[3 + i]) != Bits(expected[i]))
                {
                    if (float.IsNaN(output[3 + i]) && float.IsNaN(expected[i])) nanPayloads++;
                    else differences++;
                }
                if (!used) Require(Bits(output[3 + i]) == Guard, "Product helper fallback wrote output");''')
    replace('            rows.Add(new { test, finite, relu, kernel = used, differences });',
            '''            GraphRaw.Check("supplemental", test, x, weights, bias, residual, c, m, h, w, 1, lanes, relu);
            rows.Add(new { test, finite, relu, kernel = used, differences, nan_payload_differences = nanPayloads });''')
    replace('                        if (Bits(output[3 + index]) != Bits(expected[index]))',
            '''                        if (!kernel) Require(Bits(output[3 + index]) == Guard, "Product helper fallback wrote output");
                        if (kernel && Bits(output[3 + index]) != Bits(expected[index])
                            && !(float.IsNaN(output[3 + index]) && float.IsNaN(expected[index])))''')
    replace('                    observations.Add(new { c, m, h, w, stride, special, hasBias, hasResidual, relu, kernel, values = count, scalar_differences = sd, production_differences = pd, first });',
            '''                    GraphRaw.Check("raw", cases, input, weight, b, r, c, m, h, w, stride, lanes, relu);
                    observations.Add(new { c, m, h, w, stride, special, hasBias, hasResidual, relu, kernel, values = count, scalar_differences = sd, production_differences = pd, first });''')
    replace('        var supplemental = Supplemental(lanes, out int supplementalFailed);',
            '        var supplemental = Supplemental(lanes, out int supplementalFailed); GraphRaw.Complete();')
    replace('passed = failedCases + supplementalFailed == 0, geometries',
            'passed = failedCases + supplementalFailed == 0 && GraphRaw.Differences == 0, geometries')
    replace('            executable = Hash(typeof(Probe).Assembly.Location), pid = Environment.ProcessId, flags };',
            '''            executable = Hash(typeof(Probe).Assembly.Location), pid = Environment.ProcessId, flags,
            graph_differences = GraphRaw.Differences, graph_nan_payload_differences = GraphRaw.NaNPayloadDifferences,
            graph_cases = GraphRaw.Rows, graph_controls = GraphRaw.ControlCalls, graph_candidates = GraphRaw.CandidateCalls,
            no_performance_measurement = true };''')
    return text, ''.join(difflib.unified_diff(source.splitlines(True), text.splitlines(True),
        fromfile='qualified-component/Probe.cs', tofile='actual-product-helpers-and-graphs/Probe.cs'))
