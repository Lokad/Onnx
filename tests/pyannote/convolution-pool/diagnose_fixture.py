"""Observe the failed synthetic fixture's pool counters without changing it."""
import json
import traceback
import common

ROOT = common.ROOT
BASE = ROOT / 'artifacts/pyannote-convolution-pool-fixture-diagnostic-20260921'
SOURCE = ROOT / 'artifacts/pyannote-convolution-pool-v4-20260921'


def main():
    assert not BASE.exists()
    state = common.read(SOURCE / 'preparation.json')
    assert state['complete'] and state['code'] == 1
    common.terminal(state['supervisor'])
    for run in state['runs']:
        for pid, birth in run['members'].items():
            common.terminal(dict(pid=int(pid), birth=birth))
    common.verify(common.read(SOURCE / 'focused-prepared.json')['files'])
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    project = BASE / 'Diagnostic.csproj'
    project.write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>'''
        + str(SOURCE / 'runtime/Lokad.Onnx.dll') + '''</HintPath></Reference></ItemGroup></Project>''', encoding='utf8')
    (BASE / 'Program.cs').write_text('''using System.Reflection;
using System.Text.Json;
using Lokad.Onnx;
var assembly = Assembly.LoadFrom(args[0]);
var method = assembly.GetType("Lokad.Onnx.Backend.Tests.ConvolutionPoolTests")!.GetMethod("Graph", BindingFlags.Static | BindingFlags.NonPublic)!;
foreach (bool fused in new[] { false, true }) foreach (bool keep in new[] { false, true })
{
    var graph = (ComputationalGraph)method.Invoke(null, new object[] { fused, keep })!;
    var execution = graph.CreateExecution(ExecutionOptions.Memory);
    for (int repeat = 0; repeat < 4; repeat++)
    {
        var input = DenseTensor<float>.OfShape(1, 1, 2, 4); input.Fill(repeat + 1f);
        execution.Reset();
        bool passed = execution.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory);
        Console.WriteLine(JsonSerializer.Serialize(new { fused, keep, repeat, passed, execution.LastErrorMessage,
            execution.LastPoolAllocatedNew, execution.LastPoolAllocatedNewBytes, execution.LastPoolReusedBytes, execution.LastPoolReturned,
            lastUse = execution.LastUseIndex, inputs = execution.Inputs.Keys, outputs = execution.Outputs.Keys,
            intermediate = execution.IntermediateOutputs.ToDictionary(p => p.Key, p => p.Value?.GetType().Name) }));
    }
}
''', encoding='utf8')
    common.monitor.BASE = BASE
    owner = common.psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    try:
        flags = common.monitor.FLAGS + ['-p:NuGetAudit=false']
        for name, args in [('restore', ['dotnet', 'restore', project, *flags, '--source', common.FEED]),
                           ('build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])]:
            common.monitor.worker(state, BASE / 'state.json', name, args, BASE, [0], 8, 8, 900, True, None)
        common.save(BASE / 'prepared.json', dict(files={str(p): common.pin(p) for p in [project, BASE / 'Program.cs', BASE / 'bin/Release/net10.0/Diagnostic.dll',
            SOURCE / 'runtime/Lokad.Onnx.dll', SOURCE / 'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0/Lokad.Onnx.Backend.Tests.dll',
            common.TOOLS / 'diagnose_fixture.py', common.MONITOR]}))
        common.monitor.worker(state, BASE / 'state.json', 'run', ['dotnet', BASE / 'bin/Release/net10.0/Diagnostic.dll',
            SOURCE / 'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0/Lokad.Onnx.Backend.Tests.dll'], BASE, [0], 10, 8, 900, False, None)
        state['code'] = 0
        print((BASE / 'logs/run.log').read_text(), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        common.save(BASE / 'state.json', state)


if __name__ == '__main__':
    main()
