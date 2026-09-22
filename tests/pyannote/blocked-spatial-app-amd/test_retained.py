"""Retained product qualification must refer to the exact measured candidate."""
from pathlib import Path
import shutil
import tempfile
import unittest
from fresh_qualification import retained_callers

ROOT = Path(__file__).resolve().parents[3]


class RetainedTests(unittest.TestCase):
    def fixture(self, base):
        folder = base/'retained-product'; folder.mkdir()
        old = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v3-20260922'
        for name in ['closed.json', 'analysis.json']: shutil.copy2(old/name, folder/name)
        runtime = base/'runtimes/portable'; runtime.mkdir(parents=True)
        shutil.copy2(old/'payload/runtime/Lokad.Onnx.dll', runtime/'Lokad.Onnx.dll')
        return folder, runtime

    def test_exact_runtime_and_completed_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); _, runtime = self.fixture(base)
            self.assertTrue(retained_callers(base)['passed'])
            (runtime/'Lokad.Onnx.dll').write_bytes(b'different runtime')
            with self.assertRaises(AssertionError): retained_callers(base)

    def test_changed_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); folder, _ = self.fixture(base)
            (folder/'analysis.json').write_text('{}')
            with self.assertRaises(AssertionError): retained_callers(base)


if __name__ == '__main__': unittest.main()
