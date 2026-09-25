"""Associate complete named-method listings with the same process's load events."""
import hashlib
import re

NAMESPACE='Lokad.Onnx.Tensor`1[System.Single]'
SIGNATURES={
    'RunFloatMatMulKernel':'Lokad.Onnx.Tensor`1[float]:RunFloatMatMulKernel(int,int,int,ptr,ptr,ptr,Lokad.Onnx.TensorExecutionOptions)',
    'RunBatchedFloatMatMul':'Lokad.Onnx.Tensor`1[float]:RunBatchedFloatMatMul(Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],Lokad.Onnx.TensorExecutionOptions)',
}
TIERS={'Tier0':'QuickJitted','Instrumented Tier0':'QuickJittedInstrumented','Tier1':'OptimizedTier1','Tier1-OSR':'OptimizedTier1OSR','FullOpts':'Optimized'}
METHOD_RE=re.compile(r'^; Assembly listing for method ([^\n]+) \(([^\n]+)\)\n(.*?); Total bytes of code (\d+)[^\n]*',re.M|re.S)
BYTES_RE=re.compile(r'^\s+([0-9A-F]{2,30})\s+(.+)$',re.M)


def reconcile_listings(text,events,pid):
    text=text.replace('\r\n','\n');matches=list(METHOD_RE.finditer(text))
    assert matches and len(matches)==text.count('; Assembly listing for method ')==text.count('; Total bytes of code ')
    loads=[e for e in events if e['provider']=='Microsoft-Windows-DotNETRuntime' and e['name']=='Method/LoadVerbose'
           and e['payload']['MethodNamespace']==NAMESPACE and e['payload']['MethodName'] in SIGNATURES]
    assert loads and all(e['pid']==pid for e in loads)
    parsed=[]
    for index,match in enumerate(matches):
        method,=[name for name,signature in SIGNATURES.items() if match[1]==signature]
        assert match[2] in TIERS,match[2]
        labels=re.findall(r'^(G_M\d+_IG\d+):',match[0],re.M)
        assert labels and len(labels)==len(set(labels)) and set(re.findall(r'G_M\d+_IG\d+',match[0]))<=set(labels)
        instructions=[]
        for found in BYTES_RE.finditer(match[3]):
            raw=bytes.fromhex(found[1]);assert 0<len(raw)<=15
            instructions.append(dict(bytes=found[1],text=found[2].strip()))
        assert instructions,'Requested instruction bytes absent'
        assert all(len(x['bytes'])%2==0 for x in instructions)
        parsed.append(dict(index=index,method=method,signature=match[1],tier=match[2],runtime_tier=TIERS[match[2]],
            native_bytes=int(match[4]),listing_sha256=hashlib.sha256(match[0].encode()).hexdigest(),
            instruction_bytes=sum(len(i['bytes'])//2 for i in instructions),instructions=instructions,
            calls=[r['text'] for r in instructions if re.match(r'^(call|tail\.jmp)\s',r['text'])],
            comments=[line for line in match[3].splitlines() if line.startswith(';')],listing=match[0]))
    for method in SIGNATURES:
        emitted=[r for r in parsed if r['method']==method];observed=[e for e in loads if e['payload']['MethodName']==method]
        assert emitted and len(emitted)==len(observed),(method,len(emitted),len(observed))
        assert len({e['payload']['MethodID'] for e in observed})==1
        assert len({e['payload']['MethodSignature'] for e in observed})==1
        for version,(row,event) in enumerate(zip(emitted,observed,strict=True)):
            p=event['payload'];actual=p['OptimizationTier']
            difference=row['runtime_tier']!=actual
            if difference:
                assert method=='RunBatchedFloatMatMul' and version==0
                assert row['tier']=='Instrumented Tier0' and actual=='QuickJitted'
                assert any('CORINFO_HELP_COUNTPROFILE32' in c for c in row['calls'])
                assert any('CORINFO_HELP_PATCHPOINT' in c for c in row['calls'])
            assert row['native_bytes']==int(p['MethodSize'])
            row['runtime_tier']=actual
            row['rendered_runtime_label_difference']=difference
            row['event']=event
    assert len(parsed)==len(loads)
    return dict(passed=True,listings=len(parsed),method_loads=len(loads),rows=parsed,
                source='Fixed runtime disassembly logging; association uses same-process method identity, tier, order and size.')
