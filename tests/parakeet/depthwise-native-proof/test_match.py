import struct
import unittest
from match import locate


class CompleteByteIdentity(unittest.TestCase):
    def fixture(self):
        candidate = bytearray(range(128))
        relocations = [dict(offset=4, type=2, symbol='MlasMaskMoveAvx', addend=-4),
                       dict(offset=16, type=2, symbol='MlasMaskMoveAvx', addend=12)]
        original = bytearray(2048)
        for r in relocations: struct.pack_into('<i', candidate, r['offset'], 0)
        original[256:384] = candidate
        original[1024:1056] = struct.pack('<8I', *range(8))
        for r in relocations:
            struct.pack_into('<i', original, 256+r['offset'], 1024+r['addend']-(256+r['offset']))
        loads = [dict(offset=0, address=0x2000, size=2048, flags=5)]
        return bytes(candidate), original, relocations, loads

    def test_resolves_constants_and_nonzero_elf_load_address(self):
        candidate, original, relocations, loads = self.fixture()
        found = locate(candidate, original, relocations, loads)
        self.assertEqual((found['start'], found['address']), (256, 0x2100))
        self.assertEqual({r['symbol_address'] for r in found['relocations']}, {'0x2400'})

    def test_rejects_wrong_referenced_constant(self):
        candidate, original, relocations, loads = self.fixture()
        original[1024] ^= 1
        with self.assertRaises(AssertionError): locate(candidate, original, relocations, loads)

    def test_rejects_one_changed_instruction_byte(self):
        candidate, original, relocations, loads = self.fixture()
        original[256+2] ^= 1
        with self.assertRaises(AssertionError): locate(candidate, original, relocations, loads)

    def test_rejects_ambiguous_or_nonexecutable_match(self):
        candidate = bytes(range(128)); loads = [dict(offset=0, address=0, size=512, flags=5)]
        with self.assertRaises(AssertionError): locate(candidate, candidate+candidate+bytes(256), [], loads)
        loads[0]['flags'] = 4
        with self.assertRaises(AssertionError): locate(candidate, candidate+bytes(384), [], loads)


if __name__ == '__main__': unittest.main()
