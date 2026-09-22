"""Prove compiler renames and qualify the unchanged first candidate binary."""
import difflib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import traceback
import unittest
import xml.etree.ElementTree as ET
from normalize import normalize

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-composition-review-v2-20260922'
PRIOR = ROOT/'artifacts/pyannote-blocked-spatial-composition-20260922'
COMPONENT = ROOT/'artifacts/pyannote-vector-input-layout-20260922'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('review_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify, terminal = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal


def review():
    value = read(PRIOR/'instructions.json'); core, data = value['observations']
    result, normalized = normalize(core)
    assert len(result['renames']) == 197 and len(result['symbols']) == 217
    expected = {
        ('Lokad.Onnx.ComputationalGraph', '.ctor'), ('Lokad.Onnx.ComputationalGraph', 'InvalidatePreparation'),
        ('Lokad.Onnx.ComputationalGraph', 'RefreshLifetimeAnalysis'), ('Lokad.Onnx.ComputationalGraph', 'RunCoreInner'),
        ('Lokad.Onnx.GraphExecution', '.ctor'), ('Lokad.Onnx.GraphPacking', 'PackMatMulWeights'),
        ('Lokad.Onnx.TensorExecutionOptions', '.ctor'), ('Lokad.Onnx.TensorExecutionOptions', 'Equals'),
        ('Lokad.Onnx.TensorExecutionOptions', 'GetHashCode'), ('Lokad.Onnx.Tensor`1[T]', 'Conv2DFloatCore')}
    assert len(result['changed']) == 10 and {tuple(k.split('::')[:2]) for k in result['changed']} == expected
    assert result['unchanged'] == 3103 and len(result['added']) == 48
    for key in result['added']:
        owner, name, _ = key.split('::')
        assert owner.startswith(('Lokad.Onnx.ConvBlockedSpatial', 'Lokad.Onnx.GraphConvPacking', 'Lokad.Onnx.PackedConvWeight')) or (
            owner == 'Lokad.Onnx.Tensor`1[T]' and name in ['PlanConvBlockedScratch', 'TryConvBlockedSpatial']) or (
            owner == 'Lokad.Onnx.TensorExecutionOptions' and name in ['get_PackedConvWeights', 'set_PackedConvWeights']), key
    equal = []
    for key, body in value['component_methods'].items():
        if not key.startswith('BlockedSpatial::') or key.split('::')[1] in ['Execute', 'Run']: continue
        target = key.replace('BlockedSpatial::', 'Lokad.Onnx.ConvBlockedSpatial::')
        assert normalized[target] == json.loads(body.replace('BlockedSpatial::', 'Lokad.Onnx.ConvBlockedSpatial::')), key
        equal.append(key)
    assert len(equal) == 11
    assert core['public_surface_equal'] and data['public_surface_equal']
    assert data['unchanged_methods'] == data['methods'] == 697 and not data['differences'] and not data['removed'] and not data['added']
    assert core['after_sha256'] == pin(PRIOR/'runtime/Lokad.Onnx.dll')['sha256']
    assert data['after_sha256'] == pin(PRIOR/'runtime/Lokad.Onnx.Data.dll')['sha256']
    return dict(passed=True, core=result, component_methods_equal=equal, data_unchanged=697, public_surface_equal=True,
        original_inventory=pin(PRIOR/'instructions.json'), original_cctor_rename=core['compiler_rename'])


def suites():
    result = []; ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    for mode in ['normal', 'disabled']:
        path = BASE/'test-results'/('focused-'+mode+'.trx'); root = ET.parse(path)
        counters = root.find('.//t:Counters', ns).attrib; rows = root.findall('.//t:UnitTestResult', ns)
        assert int(counters['failed']) == 0 and int(counters['passed']) == int(counters['total']) == len(rows) == 31
        assert all(r.attrib['outcome'] == 'Passed' for r in rows)
        result.append(dict(mode=mode, tests=len(rows), names=[r.attrib['testName'] for r in rows], file=pin(path)))
    return result


def main():
    assert not BASE.exists()
    assert pin(PRIOR/'failure-closed.json')['sha256'] == 'b52ca1fe7ed0edf06094708ae581c1224cd659a1903ec6844ac58af76856b901'
    prior = read(PRIOR/'failure-closed.json'); assert prior['retained_failure'] and not prior['passed']
    for name, wanted in prior['files'].items(): assert pin(PRIOR/name) == wanted, name
    for identity in prior['identities']: terminal(identity)
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'test-results').mkdir()
    suite = unittest.defaultTestLoader.discover(str(TOOLS), pattern='test_normalize.py'); output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    save(BASE/'selftest.json', dict(passed=result.wasSuccessful(), tests=result.testsRun, output=output.getvalue()))
    assert result.wasSuccessful() and result.testsRun == 7
    save(BASE/'instruction-review.json', review())
    source = BASE/'focused'; source.mkdir()
    before = (PRIOR/'source/tests/Lokad.Onnx.Backend.Tests/ConvBlockedSpatialTests.cs').read_text()
    old = '        Equal(expected, Run(graph, inputs: new() { ["x"] = graph.Inputs["x"], ["w"] = overrideWeight }).ToArray());'
    new = '''        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = graph.Inputs["x"], ["w"] = overrideWeight },
            false, ExecutionProvider.CPU, ExecutionOptions.Default), graph.LastErrorMessage);
        Equal(expected, Assert.IsType<DenseTensor<float>>(graph.Outputs["y"]).ToArray());'''
    assert before.count(old) == 1; after = before.replace(old, new)
    (source/'ConvBlockedSpatialTests.cs').write_text(after, encoding='utf8')
    (BASE/'test-api-correction.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='first-unexecuted-test', tofile='explicit-runtime-weights')), encoding='utf8')
    project = source/'Focused.csproj'
    project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><AssemblyName>Lokad.Onnx.Backend.Tests</AssemblyName><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><IsTestProject>true</IsTestProject><IsPackable>false</IsPackable></PropertyGroup><ItemGroup><Using Include="Xunit"/><Reference Include="Lokad.Onnx"><HintPath>{PRIOR/'runtime/Lokad.Onnx.dll'}</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>{PRIOR/'runtime/Google.Protobuf.dll'}</HintPath></Reference><PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.9.0"/><PackageReference Include="xunit" Version="2.8.1"/><PackageReference Include="xunit.runner.visualstudio" Version="2.8.1"/></ItemGroup></Project>''', encoding='utf8')
    files = {p.as_posix(): pin(p) for p in [PRIOR/'failure-closed.json', MONITOR, *TOOLS.glob('*'), *source.glob('*'), BASE/'test-api-correction.diff', BASE/'instruction-review.json'] if p.is_file()}
    save(BASE/'inputs.json', dict(files=files, prior=pin(PRIOR/'failure-closed.json'), product_bytes_unchanged=True))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'controller.json'; save(path, state); flags = monitor.FLAGS+['-p:NuGetAudit=false']
    try:
        def run(name, command, numerical=False):
            monitor.worker(state, path, name, command, ROOT, [0], 12 if numerical else 8, 8, 900, True, source)
            print(name, 'passed', flush=True)
        run('restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'])
        run('build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
        assert pin(source/'bin/Release/net10.0/Lokad.Onnx.dll') == prior['core']
        for mode in ['normal', 'disabled']:
            previous = monitor.clean_env
            if mode == 'disabled': monitor.clean_env = lambda: previous() | {'DOTNET_EnableHWIntrinsic': '0'}
            try:
                run('focused-'+mode, ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                    '--logger', 'trx;LogFileName=focused-'+mode+'.trx', '--results-directory', BASE/'test-results'], True)
            finally: monitor.clean_env = previous
        for name, wanted in prior['files'].items(): assert pin(PRIOR/name) == wanted, name
        verify(files)
        save(BASE/'verified.json', dict(passed=True, files=files, suites=suites(), core=prior['core'], data=prior['data'],
            product_bytes_unchanged=True, preparation_only=True, models_qualified=False, performance_qualified=False))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': main()
