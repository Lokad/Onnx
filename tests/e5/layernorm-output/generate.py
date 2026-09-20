"""Copy the complete product LayerNorm kernel, widening only its final transform."""
from pathlib import Path
import argparse,hashlib,importlib.util,json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];extractor=root/'tests/e5/softmax-zero-blocks/generate.py'
    spec=importlib.util.spec_from_file_location('extractor',extractor);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    source=root/'src/Lokad.Onnx/TensorOps.Norm.cs';body=parser.method(source.read_text(),'LayerNormFloatInto')
    original=body.replace('static void LayerNormFloatInto','public static void Copy')
    anchor='                var vinv = new Vector<double>(inv);\n                i = 0;'
    assert body.count(anchor)==1
    wide='''
                var wideMean = Vector512.Create(mean);
                var wideInv = Vector512.Create(inv);
                for (; i <= block - 16; i += 16)
                {
                    var (x0, x1) = Vector512.Widen(Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(xs), (nuint)(off + i)));
                    var (s0, s1) = Vector512.Widen(Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(ss), (nuint)i));
                    var b0 = Vector512<double>.Zero;
                    var b1 = Vector512<double>.Zero;
                    if (bd is not null)
                        (b0, b1) = Vector512.Widen(Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(bs), (nuint)i));
                    var r0 = (x0 - wideMean) * wideInv * s0 + b0;
                    var r1 = (x1 - wideMean) * wideInv * s1 + b1;
                    Vector512.Narrow(r0, r1).CopyTo(os.Slice(off + i));
                }'''
    candidate=body.replace('static void LayerNormFloatInto','public static void WideOutput').replace(anchor,anchor+wide)
    assert candidate.replace(anchor+wide,anchor).replace('WideOutput','LayerNormFloatInto').replace('public static void','static void',1)==body
    text='using System;\nusing System.Numerics;\nusing System.Runtime.Intrinsics;\nusing System.Runtime.InteropServices;\nusing Lokad.Onnx;\nnamespace LayerNormOutput;\ninternal static class Kernels\n{\n'+original+'\n'+candidate+'\n}\n'
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'Kernels.cs';path.write_text(text,encoding='utf-8')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    with (a.output/'source.json').open('x') as f:json.dump(dict(source_sha256=sha(source),extractor_sha256=sha(extractor),generator_sha256=sha(Path(__file__)),generated_sha256=sha(path),
        scope='Entire original body plus sixteen-element final-transform loop; unchanged statistics, original vector/scalar tails and scalar path',inserted_loop=wide),f,indent=2)
    print('Generated exact original copy and final-transform-only candidate.')

if __name__=='__main__':main()
