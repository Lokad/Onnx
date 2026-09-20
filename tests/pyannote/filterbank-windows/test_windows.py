import copy, unittest
from shared import *

class Windows(unittest.TestCase):
    def test_complete_and_padded_extraction(self):
        for size in (1, 399, 400, 93680, 99893, 159999, 160000, 160001, 176000, 201573, 480000):
            source = ((np.arange(size) % 257 - 128) / 256).astype(np.float32)
            before = source.copy()
            starts = [0]
            while starts[-1] + 160000 < size: starts.append(starts[-1] + 16000)
            for index, start in enumerate(starts):
                actual, valid = extract(source, index)
                expected = np.asarray([float(source[j]) if j < size else 0. for j in range(start, start + 160000)], np.float32)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(valid, min(160000, size - start))
                actual[0] = .5
                np.testing.assert_array_equal(source, before)
            with self.assertRaises(AssertionError): extract(source, len(starts))

    def test_refusals_and_complete_inventory(self):
        for samples, index in [(np.zeros(0, np.float32), 0), (np.zeros(400, np.float64), 0),
                               (np.zeros((1, 400), np.float32), 0), (np.full(400, np.nan, np.float32), 0),
                               (np.full(400, 1.1, np.float32), 0), (np.zeros(400, np.float32), -1)]:
            with self.assertRaises(AssertionError): extract(samples, index)
        manifest = dict(cases=[dict(name='x', windows=[dict(features='x0'), dict(features='x1')]), dict(name='silence', windows=[dict(scores='s')])])
        trace = dict(reports=[dict(name='x', stage='features', reference='x0'), dict(name='x', stage='features', reference='x1')])
        self.assertEqual(len(feature_inventory(manifest, trace)), 2)
        for rows in [trace['reports'][:1], trace['reports'][::-1], trace['reports'] + trace['reports'][:1]]:
            with self.assertRaises(AssertionError): feature_inventory(manifest, dict(reports=rows))
        changed = copy.deepcopy(trace); changed['reports'][1]['reference'] = 'other'
        with self.assertRaises(AssertionError): feature_inventory(manifest, changed)

if __name__ == '__main__':
    psutil_module().Process().cpu_affinity([0])
    unittest.main()
