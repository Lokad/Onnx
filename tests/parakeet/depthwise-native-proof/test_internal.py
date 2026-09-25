import struct
import unittest
from match_internal import resolve_internal


class SameTextRelocation(unittest.TestCase):
    def test_reconstructs_internal_call_and_rejects_unrelated_symbol(self):
        text = bytearray(128)
        symbol = dict(section=1, name='MlasConvPostProcessFloatAvx512FFilter1Output1', value=96)
        resolve_internal(text, 12, 4, symbol, -4, 1)
        self.assertEqual(struct.unpack_from('<i', text, 12)[0], 80)
        with self.assertRaises(AssertionError):
            resolve_internal(text, 12, 4, dict(symbol, name='Other'), -4, 1)
        with self.assertRaises(AssertionError):
            resolve_internal(text, 12, 4, dict(symbol, section=2), -4, 1)


if __name__ == '__main__': unittest.main()
