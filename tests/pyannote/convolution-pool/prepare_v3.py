"""Check the exact v2 binary after proving one generated initializer rename."""
import common

SECOND = common.ROOT / 'artifacts/pyannote-convolution-pool-v2-20260921'
failure = SECOND / 'failure-closed.json'
assert common.pin(failure)['sha256'] == '1124373d12b1ee8ce8f30eb1ff82d8a8d1fd197da3ef2d2ea72105b55d0e4ca1'
closed = common.read(failure)
assert not closed['passed'] and not closed['inference_started']
common.verify(closed['files'])
for identity in closed['identities']:
    common.terminal(identity)
common.BASE = common.ROOT / 'artifacts/pyannote-convolution-pool-v3-20260921'
common.monitor.BASE = common.BASE
source = (common.TOOLS / 'prepare.py').read_text(encoding='utf8')


def replace(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)


replace('from modify_source import modify', 'from modify_source_v2 import modify')
replace("for name, project in [('core', source / 'src/Lokad.Onnx/Lokad.Onnx.csproj'), ('bridge', bridge / 'Bridge.csproj')]:",
    "for name, project in [('bridge', bridge / 'Bridge.csproj')]:")
replace("shutil.copy2(source / 'src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll', runtime / 'Lokad.Onnx.dll')",
    """shutil.copy2(SECOND / 'runtime/Lokad.Onnx.dll', runtime / 'Lokad.Onnx.dll')
        for path in (source / 'src/Lokad.Onnx').rglob('*.cs'):
            assert pin(path) == pin(SECOND / 'source' / path.relative_to(source)), path
        assert pin(runtime / 'Lokad.Onnx.dll') == pin(SECOND / 'runtime/Lokad.Onnx.dll')""")
replace("    path.write_text(text, encoding='utf8')\n    owner = psutil.Process()",
    """    anchor = '    var oldMethods = Inspect(oldAssembly); var newMethods = Inspect(newAssembly);'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '\\n' + (TOOLS / 'InitializerOrdinal.cs.txt').read_text(encoding='utf8'))
    text = text.replace('assembly = name, methods = oldMethods.Count,', 'assembly = name, compiler_rename = compilerRename, methods = oldMethods.Count,')
    path.write_text(text, encoding='utf8')
    owner = psutil.Process()""")
replace('files = {rel(receipt): pin(receipt), rel(MONITOR): pin(MONITOR)}',
    'files = {rel(receipt): pin(receipt), rel(MONITOR): pin(MONITOR), rel(FAILURE): pin(FAILURE)}')
namespace = dict(__name__='proven_initializer_rename', __file__=str(common.TOOLS / 'prepare.py'), FAILURE=failure, SECOND=SECOND)
exec(compile(source, namespace['__file__'], 'exec'), namespace)
namespace['main']()
