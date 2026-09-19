"""Extract default masked softmax and add an exact all-underflow early return."""
from pathlib import Path
import argparse,hashlib,json,re

def method(text,name):
    m=re.search(r'    (?:internal |public )?static [^\n]*\b'+name+r'\(',text);assert m,name
    opening=text.index('{',m.end());end=closing(text,opening)
    return text[m.start():end]

def closing(text,start):
    depth=1;end=start+1
    while depth:
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    return end

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[3]
    math=root/'src/Lokad.Onnx/MathOps.cs';tensor=root/'src/Lokad.Onnx/TensorOps.Elementwise.cs'
    ms=math.read_text(encoding='utf-8');ts=tensor.read_text(encoding='utf-8')
    exp=method(ms,'ExpVectorNonpositive').replace('ExpVectorNonpositive','OriginalExp')
    candidate=exp.replace('OriginalExp','CandidateExp');opening=candidate.index('{')+1
    candidate=candidate[:opening]+'\n        if (Vector.LessThanAll(v, new Vector<float>(-88.722839f))) return Vector<float>.Zero;'+candidate[opening:]
    body=method(ts,'SoftmaxMaskedFloatSpanPtr');start=body.index('            bool useWideExp =');end=body.index(';',start)+1
    body=body[:start]+body[end:];removed=0
    while 'if (useWideExp)' in body:
        start=body.index('if (useWideExp)');end=closing(body,body.index('{',start));body=body[:start]+body[end:];removed+=1
    assert removed==3 and body.count('MathOps.ExpVectorSoftmax(')==3
    original=body.replace('SoftmaxMaskedFloatSpanPtr','Original').replace('MathOps.ExpVectorSoftmax(','OriginalExp(')
    changed=body.replace('SoftmaxMaskedFloatSpanPtr','Candidate').replace('MathOps.ExpVectorSoftmax(','CandidateExp(')
    maximum=method(ts,'SoftmaxContiguousMaxMasked')
    adaptive='''    internal static void Adaptive(Span<float> input, Span<float> mask, Span<float> output, int rows, int columns, bool simd)
    {
        bool possible = false;
        if (mask.Length < columns) throw new ArgumentException("Short mask");
        for (int i = 0; i < columns; i++) if (mask[i] < -88.722839f) { possible = true; break; }
        if (possible) Candidate(input, mask, output, rows, columns, simd);
        else Original(input, mask, output, rows, columns, simd);
    }
'''
    source='using System;\nusing System.Numerics;\nusing System.Runtime.CompilerServices;\nnamespace ZeroBlocks;\ninternal static class Kernels\n{\n'
    source+='[MethodImpl(MethodImplOptions.AggressiveInlining)]\n'+exp+'\n[MethodImpl(MethodImplOptions.AggressiveInlining)]\n'+candidate+'\n'+maximum+'\n'+original+'\n'+changed+'\n'+adaptive+'}\n'
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'Kernels.cs';path.write_text(source,encoding='utf-8')
    pins=dict(source={str(p.relative_to(root)):sha(p) for p in (math,tensor)},generator_sha256=sha(Path(__file__)),kernel_sha256=sha(path),
        removed_unselected_wide_branches=removed,changed_exponential_call_sites=3,cutoff=-88.722839,scope='Current nonpositive-on/wide-off defaults only; actual frozen core is separately checked')
    (a.output/'source.json').write_text(json.dumps(pins,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
