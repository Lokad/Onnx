"""Retained real captures must not leak old clocks or hide a changed route."""
import copy
import unittest
from binding import bind
from review import calculate


class AttentionJoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bound = bind()
        cls.value, cls.rows = calculate(cls.bound)

    def test_old_clocks_and_copy_costs_are_irrelevant(self):
        poisoned = copy.deepcopy(self.bound['records'])
        for row in poisoned:
            row['node_ticks'] = row['interval_ticks'] = row['copy_bytes'] = float('nan')
        value, rows = calculate(self.bound, poisoned)
        self.assertEqual((value, rows), (self.value, self.rows))

    def test_missing_duplicate_changed_mapping_and_changed_runtime_shape_refuse(self):
        for mutation in ['missing', 'duplicate', 'mapping', 'shape']:
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                altered = copy.deepcopy(self.bound['records'])
                index = next(i for i, row in enumerate(altered) if '/self_attn/' in row['name'])
                if mutation == 'missing':
                    altered.pop(index)
                elif mutation == 'duplicate':
                    altered.append(copy.deepcopy(altered[index]))
                elif mutation == 'mapping':
                    altered[index]['mapped'] = not altered[index]['mapped']
                else:
                    # Keep the divisibility guards true: the fresh native shape must reject it.
                    altered[index]['m'] += 6
                    altered[index]['frames'] += 3 if '/linear_pos/' in altered[index]['name'] else 6
                calculate(self.bound, altered)


if __name__ == '__main__':
    unittest.main()
