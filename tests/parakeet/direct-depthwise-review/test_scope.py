import copy
import json
from pathlib import Path
import unittest
from scope import reconcile

ROOT = Path(__file__).resolve().parents[3]


class ScopeIdentity(unittest.TestCase):
    def setUp(self):
        self.inventory = json.loads((ROOT/'artifacts/parakeet-direct-depthwise-build-v2-amd-20260925/build-collected/logs/instructions.json').read_text())

    def test_actual_instruction_and_flag_identity(self):
        result, proof = reconcile(self.inventory)
        self.assertEqual(result['observations'][0]['unchanged_methods'], 3276)
        self.assertEqual(len(proof['renamed_methods']), 3)

    def test_rejects_changed_instruction_in_renamed_body(self):
        row = self.inventory['observations'][0]
        key = next(k for k in row['added'] if 'b__548_0' in k)
        body = json.loads(row['candidate_methods'][key]); body['instructions'][0]['operand'] = '00000040'
        row['candidate_methods'][key] = json.dumps(body)
        with self.assertRaises(AssertionError): reconcile(self.inventory)

    def test_rejects_changed_flag_and_unexpected_rename(self):
        row = self.inventory['observations'][0]
        key = next(k for k in row['added'] if 'DisplayClass547_0' in k)
        row['method_flags_after'][key] ^= 512
        with self.assertRaises(AssertionError): reconcile(self.inventory)
        self.setUp(); self.inventory['observations'][0]['removed'].append('unexpected')
        with self.assertRaises(AssertionError): reconcile(self.inventory)


if __name__ == '__main__': unittest.main()
