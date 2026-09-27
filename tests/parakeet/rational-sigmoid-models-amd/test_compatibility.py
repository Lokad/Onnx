import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from compatibility import review, reconcile, CURRENT, QUALIFIED, BUILD
from cross_numeric import compare_values, compare_native
from protocol import read


class ScopeTests(unittest.TestCase):
    def test_retained_consumer_and_product_scope(self):
        value = review()
        self.assertEqual(value['underlying_methods_reconciled'], 3979)
        self.assertFalse(value['component_screen_admitted'])
        self.assertEqual(len(value['failed_component_controls']), 13)
        self.assertEqual(len(value['failed_component_cases']), 4)

    def test_unrelated_method_or_flag_change_rejected(self):
        root = read(QUALIFIED/'collected/inventory/instructions.json')
        candidate = read(BUILD/'build-collected/logs/instructions.json')
        args = (read(CURRENT/'analysis.json')['identities']['candidate'],
                read(QUALIFIED/'analysis.json')['built'], read(BUILD/'analysis.json')['product'])
        for flag in [False, True]:
            bad = copy.deepcopy(candidate); row = bad['observations'][1]
            key = next(iter(row['normalized_methods']))
            if flag: row['method_flags_after'][key] ^= 8
            else: row['normalized_methods'][key] += 'changed'
            with self.assertRaises(AssertionError): reconcile(root, bad, *args)


class NumericComparisonTests(unittest.TestCase):
    def floats(self, values): return np.asarray(values, dtype='<f4').tobytes()

    def test_approximate_float_and_relative_scale(self):
        row = compare_values(self.floats([0., 1000.]), self.floats([0.00005, 1000.05]), 'Float', [2])
        self.assertFalse(row['bit_identical'])
        self.assertLessEqual(row['maximum_scaled_error'], 1e-4)

    def test_out_of_bound_or_nonfinite_float_rejected(self):
        for value in [0.0002, float('nan'), float('inf')]:
            with self.assertRaises(AssertionError):
                compare_values(self.floats([0.]), self.floats([value]), 'Float', [1])

    def test_integer_shape_and_size_contracts(self):
        for dtype, native in [('Int32', '<i4'), ('Int64', '<i8')]:
            left = np.asarray([123], dtype=native).tobytes()
            right = np.asarray([124], dtype=native).tobytes()
            self.assertTrue(compare_values(left, left, dtype, [1])['bit_identical'])
            with self.assertRaises(AssertionError): compare_values(left, right, dtype, [1])
            with self.assertRaises(AssertionError): compare_values(left, left, dtype, [2])
        self.assertEqual(compare_values(b'', b'', 'Float', [0,7])['values'], 0)

    def test_missing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            rows = [dict(name='case', actual={}, comparisons=[dict(label='encoder', output='value', shape=[1], dtype='Float', file='absent', sha256='bad')])]
            for role in ['selected', 'candidate']:
                folder = base/(role+'-native-512'); folder.mkdir()
                (folder/'result.json').write_text(json.dumps(dict(rows=rows if role=='selected' else [])))
            with self.assertRaises(ValueError): compare_native(base, '512')


if __name__ == '__main__': unittest.main()
