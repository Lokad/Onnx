"""Extract the exact current packed tiles and a bounded overwrite control for diagnostics."""
from pathlib import Path
import argparse, hashlib, json, re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();out=args.output
    assert not out.exists()
    source_path=root/'src/Lokad.Onnx/MathOps.PackedAvx512.cs'
    math_path=root/'src/Lokad.Onnx/MathOps.cs'
    source=source_path.read_text(encoding='utf-8')
    math=math_path.read_text(encoding='utf-8')
    def extract(name):
        start=math.index('    public unsafe static void '+name+'(')
        end=math.index('{',start)+1;depth=1
        while depth:
            if math[end]=='{':depth+=1
            elif math[end]=='}':depth-=1
            end+=1
        return math[start:end]
    original='using static Lokad.Onnx.MathOps;\n'+source.replace('namespace Lokad.Onnx;','namespace WidthProbe;').replace('public partial class MathOps','internal static partial class Kernels')
    control=source.replace('namespace Lokad.Onnx;','namespace WidthProbe;').replace('public partial class MathOps','internal static partial class GeneratedControl')
    control=control.rstrip()[:-1]+'\n'+'\n\n'.join(extract(name) for name in ('mm_unsafe_vectorized_intrinsics_2x4packed_bump','mm_unsafe_vectorized_intrinsics_3x4packed'))+'\n}\n'
    changed=[]
    def overwrite(match):
        value=match[3]
        if not (value.startswith('((Vector512<float>*)(cGroup') or re.fullmatch(r'Cpv[123]\[[0-3]\]',value)):
            return match[0]
        changed.append(match[0]);return match[1]+'Vector'+match[2]+'<float>.Zero;'
    candidate=re.sub(r'(Vector(512|256)<float> c\w+ = )([^;]+);',overwrite,control.replace('class GeneratedControl','class GeneratedOverwrite'))
    assert len(changed)==60, 'Kernel layout changed: review extraction and overwrite boundaries'
    out.mkdir(parents=True)
    files={}
    for name,text in [('Original.cs',original),('GeneratedControl.cs',control),('GeneratedOverwrite.cs',candidate)]:
        data=text.encode('utf-8')
        with (out/name).open('xb') as stream:stream.write(data)
        files[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    pins=dict(source={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (source_path,math_path)},files=files,changed_initializers=changed)
    with (out/'source.json').open('x',encoding='utf-8') as stream:json.dump(pins,stream,indent=2)


if __name__=='__main__':main()
