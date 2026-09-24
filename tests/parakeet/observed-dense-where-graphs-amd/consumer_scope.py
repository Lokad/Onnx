"""Bind full-case scoring and checks to the two already qualified protocols."""
import ast
from pathlib import Path
from protocol import pin

TOOLS=Path(__file__).resolve().parent;ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'validated-composition-graphs-amd'
E5=ROOT/'tests/benchmarks/e5-warmed-qualification-amd'


def functions(path):
    source=path.read_text(encoding='utf8')
    return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}


def verify_scope():
    files={}
    for target,source in [('protocol.py',ORIGINAL/'protocol.py'),('native.py',ORIGINAL/'native.py'),
        ('native-e5.py',E5/'native.py'),('checks_e5.py',E5/'checks.py'),
        ('statistics_base.py',ORIGINAL/'statistics.py'),('statistics_e5.py',E5/'statistics.py'),
        ('test_inventory.py',ORIGINAL/'test_inventory.py')]:
        assert (TOOLS/target).read_bytes()==source.read_bytes(),target
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    for target,source,module in [('test_statistics.py',ORIGINAL/'test_statistics.py','statistics_base'),
        ('test_e5_statistics.py',E5/'test_statistics.py','statistics_e5')]:
        expected=source.read_text(encoding='utf8').replace('from statistics import','from '+module+' import')
        assert (TOOLS/target).read_text(encoding='utf8')==expected,target
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    before,after=functions(ORIGINAL/'checks.py'),functions(TOOLS/'checks.py')
    assert before.keys()==after.keys()
    assert before['normalized_body']==after['normalized_body'] and before['consumer_inventory']==after['consumer_inventory']
    expected=before['check_result']
    expected=expected.replace("    if role!='ort':\n","    consumer_key='e5_consumer' if key=='e5-30tok' else 'consumer'\n    native_script='native-e5.py' if key=='e5-30tok' else 'native.py'\n    warmup=1200 if key=='e5-30tok' else 600\n    if role!='ort':\n")
    expected=expected.replace("read(base/'built.json')['consumer']['sha256']","read(base/'built.json')[consumer_key]['sha256']")
    expected=expected.replace("pin(base/'tools/native.py')['sha256']","pin(base/'tools'/native_script)['sha256']")
    expected=expected.replace("calls=3 if mode=='verify' else 780","calls=3 if mode=='verify' else warmup+180").replace('index<600','index<warmup')
    assert after['check_result']==expected
    before,after=functions(ORIGINAL/'remote.py'),functions(TOOLS/'remote.py')
    assert before.keys()==after.keys()
    for name in before:
        expected=before[name]
        if name=='command_for':
            old="    prefix=[sys.executable,'-B',BASE/'tools/native.py'] if role=='ort' else [DOTNET,BASE/'runtimes'/role/'ReleaseBenchmark.dll']"
            new="    native='native-e5.py' if case=='e5-30tok' else 'native.py'\n    runtimes='runtimes-e5' if case=='e5-30tok' else 'runtimes'\n    prefix=[sys.executable,'-B',BASE/'tools'/native] if role=='ort' else [DOTNET,BASE/runtimes/role/'ReleaseBenchmark.dll']"
            assert expected.count(old)==1;expected=expected.replace(old,new)
        assert after[name]==expected,name
    expected=(ORIGINAL/'audit.py').read_text(encoding='utf8')
    expected=expected.replace('from checks import check_result,consumer_inventory','from checks import check_result,consumer_inventory,e5_consumer_inventory')
    expected=expected.replace("if name.startswith('runtimes/'):","if name.startswith(('runtimes/','runtimes-e5/')):")
    needle="    assert consumer['consumer'] == built['consumer']\n"
    expected=expected.replace(needle,needle+"    e5_consumer=e5_consumer_inventory(read(c/'evidence/e5-consumer/instructions.json'),dict(previous_consumer=spec['consumer']),dict(consumer=built['e5_consumer']))\n    assert e5_consumer==read(c/'evidence/e5-consumer/review.json')\n    assert built['e5_consumer']==read(c/'stage.json')['e5_consumer']==spec['e5_consumer']\n    assert built['e5_original_receipt']==pin(c/'evidence/e5-consumer/built.json')\n    for role in ['current','candidate']:\n        assert pin(c/'runtimes-e5'/role/'Lokad.Onnx.dll')==spec['products'][role]['Lokad.Onnx.dll']\n")
    expected=expected.replace('len(clocks)==37512','len(clocks)==41112').replace('resources=resources,consumer=consumer,clocks=','resources=resources,consumer=consumer,e5_consumer=e5_consumer,clocks=')
    assert (TOOLS/'audit.py').read_text(encoding='utf8')==expected
    for name in ['checks.py','remote.py','audit.py']:files[(ORIGINAL/name).relative_to(ROOT).as_posix()]=pin(ORIGINAL/name)
    return files


if __name__=='__main__':print(verify_scope())
