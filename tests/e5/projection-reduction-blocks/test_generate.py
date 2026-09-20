import unittest
from generate import generate, ROOT


class GenerationTests(unittest.TestCase):
    def test_only_loop_count_changes_in_leaves(self):
        source = (ROOT / 'src/Lokad.Onnx/MathOps.PackedAvx512.cs').read_text(encoding='utf-8')
        files = generate(source); marker = '    [MethodImpl(MethodImplOptions.AggressiveOptimization)]\n    internal static unsafe void PackedTile12('
        original = files['Original.cs'].split(marker)[1]
        changed = files['Blocked.cs'].split(marker)[1]
        restored = changed.replace('int kb, int count)', 'int kb)').replace('j < count;', 'j < N;')
        self.assertEqual(original, restored)
        self.assertNotIn('ArrayPool', files['Blocked.cs'])
        self.assertNotIn('CopyTo', files['Blocked.cs'])

    def test_source_drift_is_refused(self):
        source = (ROOT / 'src/Lokad.Onnx/MathOps.PackedAvx512.cs').read_text(encoding='utf-8')
        for old, new in [('int kb)', 'int col)'), ('j < N;', 'j <= N;'), ('A + i * N', 'A + i * stride'),
                         ('namespace Lokad.Onnx;', 'namespace Other;')]:
            with self.subTest(old=old), self.assertRaises(AssertionError): generate(source.replace(old, new))


if __name__ == '__main__': unittest.main()
