"""Split reduction count from row stride, preserving original packed weights."""
from pathlib import Path
import argparse, hashlib, json, shutil

ROOT = Path(__file__).resolve().parents[3]


def generate(source):
    assert source.count('public partial class MathOps') == 1
    assert source.count('namespace Lokad.Onnx;') == 1
    original = ('using static Lokad.Onnx.MathOps;\n' + source
                .replace('namespace Lokad.Onnx;', 'namespace ReductionProbe;')
                .replace('public partial class MathOps', 'internal static class Original'))
    marker = '    [MethodImpl(MethodImplOptions.AggressiveOptimization)]\n    internal static unsafe void PackedTile12('
    assert original.count(marker) == 1
    leaves = original[original.index(marker):]
    assert leaves.count('int kb)') == 2
    assert leaves.count('for (int j = 0; j < N; ++j)') == 2
    leaves = leaves.replace('int kb)', 'int kb, int count)')
    leaves = leaves.replace('for (int j = 0; j < N; ++j)', 'for (int j = 0; j < count; ++j)')
    # N still determines every row address; only the loop bound changes.
    assert leaves.count('A + i * N') == 2
    prefix = original[:original.index('    internal static unsafe bool TryPackedAvx512Rows(')]
    prefix = prefix.replace('class Original', 'class Blocked')
    dispatch = '''    internal static unsafe bool Run(int m, int n, int k, float* x, float* packed, float* dest, int block)
    {
        if (!Avx512F.IsSupported || !Fma.IsSupported || m < 8 || n <= 0 || k <= 0 || k % 32 != 0
            || (block != 128 && block != 256)) return false;
        int main = m / 12 * 12;
        int rest = m - main;
        if (rest == 1) { main -= 12; rest = 13; }
        if (rest == 4 && main >= 12) { main -= 12; rest = 16; }
        else if (rest == 2 && main >= 12) { main -= 12; rest = 14; }
        int eights = 0;
        while (rest >= 8 && rest != 9) { eights += 8; rest -= 8; }
        for (int column = 0; column < k; column += 32)
            for (int offset = 0; offset < n; offset += block)
            {
                int count = System.Math.Min(block, n - offset);
                float* panel = packed + column * n + offset * 32;
                if (main > 0) PackedTile12(main, n, x + offset, panel, dest, k, column, count);
                if (eights > 0) PackedTile8(eights, n, x + main * n + offset, panel, dest + main * k, k, column, count);
            }
        x += (main + eights) * n;
        dest += (main + eights) * k;
        if (rest == 0) return true;
        if (rest % 3 == 0)
            mm_unsafe_vectorized_intrinsics_3x4packed(rest, n, k, x, packed, dest);
        else if (rest % 2 == 0)
            mm_unsafe_vectorized_intrinsics_2x4packed_bump(rest, n, k, x, packed, dest);
        else
        {
            mm_unsafe_vectorized_intrinsics_3x4packed(3, n, k, x, packed, dest);
            mm_unsafe_vectorized_intrinsics_2x4packed_bump(rest - 3, n, k, x + 3 * n, packed, dest + 3 * k);
        }
        return true;
    }

'''
    return {'Original.cs': original, 'Blocked.cs': prefix + dispatch + leaves}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'src/Lokad.Onnx/MathOps.PackedAvx512.cs'
    for name, text in generate(source.read_text(encoding='utf-8')).items():
        (args.output / name).write_text(text, encoding='utf-8', newline='\n')
    for name in ['Program.cs', 'Probe.csproj']: shutil.copyfile(Path(__file__).parent / name, args.output / name)
    def pin(path): return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result = dict(sources={p.relative_to(ROOT).as_posix(): pin(p) for p in [source, Path(__file__)]},
                  files={p.name: pin(p) for p in args.output.iterdir()})
    with (args.output / 'source.json').open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__': main()
