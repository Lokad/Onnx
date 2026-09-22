"""Apply the documented test API fixes and remove forbidden optional parameters."""
import difflib


def transform(corrected):
    source = corrected

    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)

    replace('    static float[] Values(int count, int seed = 0)',
            '    static float[] Values(int count) => Values(count, 0);\n\n    static float[] Values(int count, int seed)')
    replace('''    static ComputationalGraph Graph(int c = 16, int m = 32, int h = 3, int w = 11, int stride = 1,
        long budget = long.MaxValue, bool relu = false)''', '''    static ComputationalGraph Graph() => Graph(16, 32, 3, 11, 1, long.MaxValue, false);
    static ComputationalGraph Graph(long budget) => Graph(16, 32, 3, 11, 1, budget, false);
    static ComputationalGraph Graph(int m, long budget) => Graph(16, m, 3, 11, 1, budget, false);
    static ComputationalGraph Graph(bool relu) => Graph(16, 32, 3, 11, 1, long.MaxValue, relu);
    static ComputationalGraph Graph(int c, int stride) => Graph(c, 32, 3, 11, stride, long.MaxValue, false);

    static ComputationalGraph Graph(int c, int m, int h, int w, int stride, long budget, bool relu)''')
    replace('Graph(c, m, h, w, stride, relu: relu)', 'Graph(c, m, h, w, stride, long.MaxValue, relu)')
    replace('    static float[] Reference(ComputationalGraph graph, TensorExecutionOptions? options = null, DenseTensor<float>? weight = null)',
            '''    static float[] Reference(ComputationalGraph graph) => Reference(graph, null, null);
    static float[] Reference(ComputationalGraph graph, TensorExecutionOptions options) => Reference(graph, options, null);
    static float[] Reference(ComputationalGraph graph, DenseTensor<float> weight) => Reference(graph, null, weight);

    static float[] Reference(ComputationalGraph graph, TensorExecutionOptions? options, DenseTensor<float>? weight)''')
    replace('    static DenseTensor<float> Run(ComputationalGraph graph, ExecutionOptions? options = null, Dictionary<string, ITensor>? inputs = null)',
            '''    static DenseTensor<float> Run(ComputationalGraph graph) => Run(graph, ExecutionOptions.Default);

    static DenseTensor<float> Run(ComputationalGraph graph, ExecutionOptions options)''')
    replace('graph.Execute(inputs ?? new() { ["x"] = graph.Inputs["x"] }, true, ExecutionProvider.CPU, options ?? ExecutionOptions.Default)',
            'graph.Execute(new() { ["x"] = graph.Inputs["x"] }, true, ExecutionProvider.CPU, options)')
    replace('    static long Bytes(int c = 16, int m = 32)',
            '    static long Bytes() => Bytes(16, 32);\n\n    static long Bytes(int c, int m)')
    return source, ''.join(difflib.unified_diff(corrected.splitlines(True), source.splitlines(True), fromfile='corrected-api-consumer', tofile='normal-source-without-optional-parameters'))
