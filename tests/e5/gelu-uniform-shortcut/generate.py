"""Generate exact original and all-small shortcut arithmetic from the product source."""
from pathlib import Path
import argparse,hashlib,json,shutil

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];source=root/'src/Lokad.Onnx/MathOps.ErfInline.cs'
    text=source.read_text(encoding='utf-8');start=text.index('        var negZero');body=text[start:text.index('\n    }',start)]
    marker='        var big = Vector.GreaterThan(ax, new Vector<float>(0.921875f));'
    assert body.count(marker)==1
    shortcut=body.replace(marker,marker+'\n        if (Vector.EqualsAll(big, Vector<int>.Zero)) return Vector.BitwiseOr(rs, signBits);')
    output='''using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using Lokad.Onnx;

public delegate void Kernel(ReadOnlySpan<float> x, ReadOnlySpan<float> bias, Span<float> y);
public static class Kernels
{
    public static readonly Kernel Product=(typeof(Tensor<float>).GetMethod("BiasGeluSpanFloatInline",BindingFlags.NonPublic|BindingFlags.Static) ?? throw new Exception("Missing product reference")).CreateDelegate<Kernel>();
'''
    for name,method in [('CopyA',body),('CopyB',body),('Conditional',shortcut)]:
        output+='''    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector<float> ErfNAME(Vector<float> v)
    {
BODY
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
    public static unsafe void NAME(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w=Vector<float>.Count,m=bias.Length;
        if(w!=8 || m<=1 || m%w!=0 || xs.Length!=ys.Length || xs.Length%m!=0){Product(xs,bias,ys);return;}
        var half=new Vector<float>(.5f);var one=Vector<float>.One;var scale=new Vector<float>(.7071067811865476f);
        fixed(float* px=xs,pb=bias,py=ys)
        {
            var x=(Vector<float>*)px;var b=(Vector<float>*)pb;var y=(Vector<float>*)py;
            int n=xs.Length/w,bn=m/w,bi=0;
            for(int i=0;i<n;i++)
            {
                var tv=x[i]+b[bi];y[i]=half*tv*(one+ErfNAME(scale*tv));
                if(++bi>=bn)bi=0;
            }
        }
    }
'''.replace('NAME',name).replace('BODY',method)
    output+='}\n'
    assert not a.output.exists();a.output.mkdir(parents=True)
    (a.output/'Kernels.cs').write_text(output,encoding='utf-8',newline='\n')
    for name in ['Program.cs','Probe.csproj']:shutil.copyfile(Path(__file__).with_name(name),a.output/name)
    def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    provenance=dict(source=pin(source),generator=pin(Path(__file__)),files={p.name:pin(p) for p in a.output.iterdir() if p.is_file()})
    (a.output/'source.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Generated two exact copies and one conditional body.')

if __name__=='__main__':main()
