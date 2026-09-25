"""Fix the two source-policy failures without changing any arithmetic or guard."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
FAILED = ROOT/'artifacts/parakeet-owned-batch-isolation-root-amd-20260925'
OUT = ROOT/'artifacts/parakeet-owned-batch-isolation-root-policy-20260925'
OWNED = 'tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs'
DEPTHWISE = 'tests/Lokad.Onnx.Backend.Tests/DirectDepthwiseTests.cs'
GLOBAL = 'src/Lokad.Onnx/Global.cs'
GRAPH = 'src/Lokad.Onnx/GraphOwnedPacking.cs'
CHANGED = [GLOBAL,GRAPH,OWNED,DEPTHWISE]


def pin(path):
    with path.open('rb') as stream: return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def split_arguments(text):
    parts=[]; start=0; stack=[]
    for i,c in enumerate(text):
        if c in '([{': stack.append(c)
        elif c in ')]}':
            assert stack and stack.pop() == {')':'(',']':'[','}':'{'}[c]
        elif c == ',' and not stack: parts.append(text[start:i].strip()); start=i+1
    assert not stack
    tail=text[start:].strip()
    if tail: parts.append(tail)
    return parts


def expand_calls(text, name, parameters, defaults, evidence):
    start=text.index('    [')
    matches=list(re.finditer(r'\b'+name+r'\(',text[start:]))
    for match in reversed(matches):
        left=start+match.end(); right=left; depth=1
        while depth:
            if text[right]=='(': depth+=1
            elif text[right]==')': depth-=1
            right+=1
        arguments=split_arguments(text[left:right-1]); supplied={}
        for i,argument in enumerate(arguments):
            named=re.match(r'^(\w+)\s*:\s*(.*)$',argument,re.S)
            key,value=(named.group(1),named.group(2)) if named else (parameters[i],argument)
            assert key in parameters and key not in supplied
            supplied[key]=value
        complete=[supplied.get(key,defaults.get(key)) for key in parameters]
        assert all(value is not None for value in complete)
        evidence.append(dict(helper=name,original=arguments,explicit=complete))
        text=text[:left]+', '.join(complete)+text[right-1:]
    return text


def corrected_sources():
    result={}; calls=[]
    for name in CHANGED:
        raw=(FAILED/'bundle/source'/name).read_bytes(); text=raw.decode('utf8')
        if name==GLOBAL:
            line='[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Lokad.Onnx.Data")]'
            assert text.count(line)==1
            text=text.replace(line+'\r\n','').replace(line+'\n','')
            assert 'InternalsVisibleTo("Lokad.Onnx.Data")' not in text
        elif name==GRAPH:
            old='''    /// <summary>Consumes eligible unmapped constants of a private, newly loaded encoder.</summary>
    /// <remarks>Call before creating execution contexts. Public/default preparation never opts in.</remarks>
    internal int PrepareOwnedMatMulWeights()'''
            new='''    /// <summary>Replaces eligible float matrix initializers with independently owned packed storage.</summary>
    /// <returns>The number of initializers replaced; zero when no eligible weight or hardware is available.</returns>
    /// <remarks>
    /// Explicit opt-in for the supported Parakeet feed-forward matrix shapes and node names.
    /// Call while holding exclusive access to a newly loaded graph, before creating execution contexts.
    /// Ordinary Prepare calls never opt in. Logical values and previously held source arrays are
    /// preserved; the graph replaces each eligible initializer and can release its old reference.
    /// Shared, visible, captured and already prepared weights remain untouched. Repeated calls are
    /// idempotent. This replaces initializer storage rather than adding a retained weight-cache entry.
    /// If a later allocation fails, completed replacements leave the graph executable.
    /// </remarks>
    /// <exception cref="InvalidOperationException">The graph is executing when preparation is attempted.</exception>
    public int PrepareOwnedMatMulWeights()'''
            assert text.count(old)==1; text=text.replace(old,new)
        elif name==OWNED:
            old='Graph(int n = 4096, int k = 1024, int m = 48, long budget = 0, bool batched = false)'
            assert text.count(old)==1
            text=text.replace(old,'Graph(int n, int k, int m, long budget, bool batched)')
            text=expand_calls(text,'Graph',['n','k','m','budget','batched'],
                dict(n='4096',k='1024',m='48',budget='0',batched='false'),calls)
        else:
            for old,new in [('Data(int[] shape, uint seed = 8171)','Data(int[] shape, uint seed)'),
                ('int[]? dilation = null, TensorExecutionOptions? selected = null,','int[]? dilation, TensorExecutionOptions? selected,'),
                ('bool expectDirect = true)','bool expectDirect)')]:
                assert text.count(old)==1; text=text.replace(old,new)
            text=expand_calls(text,'Data',['shape','seed'],dict(seed='8171'),calls)
            text=expand_calls(text,'Check',['x','w','b','group','pads','strides','dilation','selected','expectDirect'],
                dict(dilation='null',selected='null',expectDirect='true'),calls)
        result[name]=text.encode('utf8')
    assert len([c for c in calls if c['helper']=='Graph'])==8
    return result,calls


def main():
    assert not OUT.exists()
    proof=read(FAILED/'closed.json')
    assert not proof['passed'] and proof['preserved_failure']
    assert pin(FAILED/'closed.json')['sha256']=='f129f7d80bf1013d39bcd69778244803e2547c57b2c03d01f971d4f344ec5075'
    for name,wanted in proof['files'].items(): assert pin(FAILED/name)==wanted,name
    applied=read(FAILED/'bundle/evidence/root-applied.json')
    for name,wanted in applied['source_files'].items(): assert pin(ROOT/name)==wanted,name
    sources,calls=corrected_sources(); OUT.mkdir()
    for name,data in sources.items():
        before=OUT/'before'/name; before.parent.mkdir(parents=True,exist_ok=True); before.write_bytes((ROOT/name).read_bytes())
        after=OUT/'source'/name; after.parent.mkdir(parents=True,exist_ok=True); after.write_bytes(data)
    receipt=dict(passed=True,failed_root=pin(FAILED/'closed.json'),before={n:pin(ROOT/n) for n in CHANGED},
        after={n:pin(OUT/'source'/n) for n in CHANGED},calls=calls,
        product_delta='Remove Data friendship and expose the existing preparation method; retain every method body.',
        test_delta='Replace optional declarations and all callers with explicit identical argument values.',
        policy_guards_unchanged=True,generator=pin(Path(__file__)))
    (OUT/'intended.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    for name,data in sources.items():
        target=ROOT/name; temporary=target.with_suffix(target.suffix+'.policytmp'); assert not temporary.exists()
        temporary.write_bytes(data); temporary.replace(target)
        assert pin(target)==receipt['after'][name]
    (OUT/'applied.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,applied=pin(OUT/'applied.json'),changed=CHANGED,calls=len(calls))))


if __name__=='__main__': main()
