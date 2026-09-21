"""Add the original injected failure to the separately observed fixture."""
from pathlib import Path
path = Path(__file__).with_name('diagnose_fixture.py')
source = path.read_text(encoding='utf8')
source = source.replace('pyannote-convolution-pool-fixture-diagnostic-20260921', 'pyannote-convolution-pool-failure-diagnostic-20260921')
anchor = '            intermediate = execution.IntermediateOutputs.ToDictionary(p => p.Key, p => p.Value?.GetType().Name) }));'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + '''
        if (repeat == 1)
        {
            execution.Reset();
            var shape = (DenseTensor<long>)graph.Initializers["shape"];
            shape.Buffer.Span[0] = 3;
            try
            {
                bool unexpectedSuccess = execution.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory);
                Console.WriteLine(JsonSerializer.Serialize(new { fused, keep, injected_failure = true, unexpectedSuccess,
                    execution.LastErrorMessage, execution.LastPoolReusedBytes, execution.LastPoolReturned }));
            }
            finally { shape.Buffer.Span[0] = 1; }
        }''')
exec(compile(source, str(path), 'exec'), dict(__name__='__main__', __file__=str(path)))
