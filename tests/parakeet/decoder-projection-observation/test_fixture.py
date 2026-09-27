"""Verify original feed reconstruction and refuse changed retained evidence."""
import copy
import json
import struct
import unittest
from fixture import BASE, OUTPUTS, construct, identity, inspect


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, cls.blobs = inspect()
        cls.reference = json.loads((BASE/'collected/parakeet-reference/manifest.json').read_text())
        cls.selected = json.loads((BASE/'collected/candidate-native-512/result.json').read_text())

    @staticmethod
    def load(item):
        return (BASE/'collected/candidate-native-512/result.json.tensors'/item['file']).read_bytes()

    def test_original_frame_and_complete_outputs(self):
        spec, blobs = self.spec, self.blobs
        original = self.load(spec['source_encoder'])
        actual = blobs[spec['inputs']['encoder_outputs']['file']]
        source_values = struct.unpack('<75776I', original)
        self.assertEqual(struct.unpack('<1024I', actual), tuple(source_values[i * 74] for i in range(1024)))
        self.assertEqual(struct.unpack('<i', blobs[spec['inputs']['targets']['file']]), (8192,))
        self.assertEqual(struct.unpack('<i', blobs[spec['inputs']['target_length']['file']]), (1,))
        for name in ['input_states_1', 'input_states_2']:
            self.assertEqual(blobs[spec['inputs'][name]['file']], bytes(5120))
        row, = [r for r in self.selected['rows'] if r['name'] == 'english-16k']
        expected = {v['output']: v for v in row['comparisons'] if v['label'] == 'step-0'}
        self.assertEqual(set(spec['outputs']), set(expected))
        for name in OUTPUTS:
            self.assertEqual(blobs[spec['outputs'][name]['file']], self.load(expected[name]))
        self.assertEqual(sum(map(len, blobs.values())), 57380)
        self.assertNotIn('core_sha256', spec)

    def test_changed_product_rejected(self):
        for field in ['core_sha256', 'data_sha256']:
            selected = copy.deepcopy(self.selected)
            selected[field] = '0' * 64
            with self.assertRaises(AssertionError):
                construct(self.reference, selected, self.load)

    def test_noninitial_state_or_frame_rejected(self):
        for field, value in [('frame', 1), ('target', 10), ('state_input_sha256', ['0' * 64] * 2)]:
            reference = copy.deepcopy(self.reference)
            reference['cases'][0]['steps'][0][field] = value
            with self.assertRaises(AssertionError):
                construct(reference, self.selected, self.load)

    def test_changed_tensor_bytes_rejected(self):
        def corrupt(item):
            raw = self.load(item)
            return bytes([raw[0] ^ 1]) + raw[1:]
        with self.assertRaises(AssertionError):
            construct(self.reference, self.selected, corrupt)

    def test_changed_output_shape_or_dtype_rejected(self):
        for field, value in [('shape', [8198]), ('dtype', 'Int32')]:
            selected = copy.deepcopy(self.selected)
            row = selected['rows'][0]
            output, = [v for v in row['comparisons'] if (v['label'], v['output']) == ('step-0', 'outputs')]
            output[field] = value
            with self.assertRaises(AssertionError):
                construct(self.reference, selected, self.load)

    def test_ambiguous_comparison_rejected(self):
        selected = copy.deepcopy(self.selected)
        selected['rows'][0]['comparisons'].append(copy.deepcopy(selected['rows'][0]['comparisons'][0]))
        with self.assertRaises(AssertionError):
            construct(self.reference, selected, self.load)

    def test_nonfinite_frame_rejected_even_with_matching_descriptor(self):
        selected = copy.deepcopy(self.selected)
        item, = [v for v in selected['rows'][0]['comparisons'] if (v['label'], v['output']) == ('encoder', 'outputs')]
        changed = struct.pack('<I', 0x7fc00000) + self.load(item)[4:]
        item['sha256'] = identity(changed)['sha256']
        with self.assertRaises(AssertionError):
            construct(self.reference, selected, lambda v: changed if v is item else self.load(v))


if __name__ == '__main__':
    unittest.main()
