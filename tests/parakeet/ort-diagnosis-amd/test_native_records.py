"""Exercise the raw-record cases that could silently bias native attribution."""
from pathlib import Path
import struct
import tempfile
import unittest
from audit_native import raw_records, samples


class NativeRecords(unittest.TestCase):
    def test_unwound_stack_optional_and_auxiliary_thread_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'perf.script'
            path.write_text(' 100/100 1.000000000: 5000\n'
                            ' 100/102 1.005000000: 5001\n'
                            ' 7fff1234 (/lib/native.so)\n', encoding='utf8')
            rows = list(samples(path))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['frames'], [])
            self.assertEqual(rows[1]['tid'], 102)
            self.assertEqual(sum(row['period'] for row in rows), 10001)

    def test_raw_ips_and_both_lost_record_kinds(self):
        def record(kind, payload):
            return struct.pack('<IHH', kind, 0, len(payload)+8)+payload
        records = record(9, struct.pack('<QIIQ', 0x1234, 100, 102, 1005000000))
        records += record(2, struct.pack('<QQ', 77, 3))
        records += record(13, struct.pack('<Q', 4))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'perf.data'
            path.write_bytes(struct.pack('<9Q', int.from_bytes(b'PERFILE2','little'),
                                         72, 0, 0, 0, 72, len(records), 0, 0)+records)
            summary, addresses = raw_records(path)
            self.assertEqual(summary['lost'], 7)
            self.assertEqual(addresses, {(100, 102, 1005000000): 0x1234})
            # A truncated raw stream must not produce apparently complete accounting.
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(AssertionError): raw_records(path)


if __name__ == '__main__': unittest.main()
