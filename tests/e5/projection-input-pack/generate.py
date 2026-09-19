"""Derive one activation-layout prototype from exact current projection source."""
from pathlib import Path
import argparse,hashlib,json,re,shutil

def generate(source):
    original='using static Lokad.Onnx.MathOps;\n'+source.replace('namespace Lokad.Onnx;','namespace WidthProbe;').replace('public partial class MathOps','internal static partial class Kernels')
    candidate=original.replace('class Kernels','class InputPacked')
    old='''        if (main > 0)
            for (int column = 0; column < k; column += 32)
                PackedTile12(main, n, x, packed + column * n, dest, k, column);'''
    new='''        if (main > 0)
        {
            var scratch = System.Buffers.ArrayPool<float>.Shared.Rent(checked(main * n));
            try
            {
                fixed (float* input = scratch)
                {
                    PackRows12(main, n, x, input);
                    for (int column = 0; column < k; column += 32)
                        PackedTile12(main, n, input, packed + column * n, dest, k, column);
                }
            }
            finally { System.Buffers.ArrayPool<float>.Shared.Return(scratch, clearArray: false); }
        }'''
    assert candidate.count(old)==1
    candidate=candidate.replace(old,new)
    start=candidate.index('                var Ap1 = A + i * N;')
    end=candidate.index('                // C addresses',start)
    assert 'var ApC = ApB + N;' in candidate[start:end]
    candidate=candidate[:start]+'                var input = A + i * N;\n'+candidate[end:]
    for row,name in enumerate(['Ap1','Ap2','Ap3','Ap4','Ap5','Ap6','Ap7','Ap8','Ap9','ApA','ApB','ApC']):
        old=f'Vector512.Create({name}[j])'
        assert candidate.count(old)==1
        candidate=candidate.replace(old,f'Vector512.Create(input[j * 12 + {row}])')
    method='''
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void PackRows12(int rows, int reduction, float* source, float* destination)
    {
        for (int i = 0; i < rows; i += 12)
        {
            float* a = source + i * reduction;
            float* p = destination + i * reduction;
            for (int j = 0; j < reduction; ++j)
            {
'''+''.join(f'                p[j * 12 + {r}] = a[{r} * reduction + j];\n' for r in range(12))+'''            }
        }
    }
'''
    candidate=candidate.rstrip()[:-1]+method+'}\n'
    assert 'Ap1[' not in candidate and candidate.count('input[j * 12 + ')==12
    return {'Original.cs':original,'InputPacked.cs':candidate}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[3]
    assert not a.output.exists()
    source=root/'src/Lokad.Onnx/MathOps.PackedAvx512.cs'
    files=generate(source.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True)
    for name,text in files.items():(a.output/name).write_text(text,encoding='utf-8',newline='\n')
    for name in ('Program.cs','Probe.csproj'):shutil.copyfile(Path(__file__).parent/name,a.output/name)
    pin=lambda path:dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    manifest=dict(source={source.relative_to(root).as_posix():pin(source)},generator=pin(Path(__file__)),files={f.name:pin(f) for f in a.output.iterdir()})
    (a.output/'source.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest))

if __name__=='__main__':main()
