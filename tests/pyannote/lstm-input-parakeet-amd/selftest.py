"""Exercise complete AMD native fixtures and reject changed or missing evidence."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
import prepare  # Adds the established local NumPy path; runs no model.
from protocol import read, save
from checks import qualify, exact_native, manifest_with_raw_hashes
from public_audit import validate_worker, refusal_checks

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT/'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922/collected/campaign'
PAYLOAD = prepare.APP_PAYLOAD


class Checks(unittest.TestCase):
    def test_native_all_arrays_and_exact_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shutil.copytree(PAYLOAD/'parakeet-reference', base/'parakeet-reference')
            result = read(APP/'portable-parakeet.json')
            pair = {'Lokad.Onnx.dll': dict(sha256=result['core_sha256']), 'Lokad.Onnx.Data.dll': dict(sha256=result['data_sha256'])}
            spec = dict(identities=dict(selected=pair, candidate=pair), consumers=dict(TranscribeReplay=dict(sha256=result['runner_sha256'])))
            for role in ['selected', 'candidate']:
                target = base/(role+'-native'); target.mkdir()
                shutil.copy2(APP/'portable-parakeet.json', target/'result.json')
                shutil.copytree(APP/'portable-parakeet.json.tensors', target/'result.json.tensors')
                self.assertTrue(qualify(base, role+'-native', spec)['passed'])
            damaged = copy.deepcopy(result); damaged['rows'][0]['comparisons'].pop()
            save(base/'candidate-native/result.json', damaged)
            with self.assertRaises(AssertionError): qualify(base, 'candidate-native', spec)
            save(base/'candidate-native/result.json', result)
            row = next(r for r in result['rows'][0]['comparisons'] if r['dtype'] == 'Float')
            path = base/'candidate-native/result.json.tensors'/row['file']
            content = bytearray(path.read_bytes()); content[0] ^= 1; path.write_bytes(content)
            with self.assertRaises(AssertionError): exact_native(base)
            with self.assertRaises(AssertionError): qualify(base, 'candidate-native', spec)

    def test_public_complete_results_and_ownership(self):
        # Reuse a genuine preceding public run, without changing its evidence.
        manifest = manifest_with_raw_hashes(PAYLOAD, 'portable')
        result = read(ROOT/'artifacts/pyannote-blocked-spatial-parakeet-20260922/candidate-public/output/result.json')
        validate_worker(result, manifest, 'conformance')
        refusal_checks(result, manifest, 'conformance')


if __name__ == '__main__': unittest.main()
