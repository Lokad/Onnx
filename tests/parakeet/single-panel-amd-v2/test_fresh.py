"""Reject missing cases, altered finite output and invalid NaN diagnostics."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import fresh_qualification as checks
from candidate_protocol import read

ROOT = Path(__file__).resolve().parents[3]
ACTUAL = ROOT / 'artifacts/pyannote-single-panel-composition-20260922'


class CallerTests(unittest.TestCase):
    def invoke(self, mutate):
        result = read(ACTUAL / 'caller-normal.json')
        # Adapt only metadata for auditor tests; retained payload differences remain diagnostics.
        result.update(runtime='10.0.8', candidate=checks.IDENTITIES['portable'][0], baseline=checks.IDENTITIES['production'][0])
        mutate(result)
        shapes = read(ROOT / 'artifacts/pyannote-two-column-20260922/payload/shapes.json')
        state = dict(runs=[dict(name='caller-normal', code=0, child=dict(pid=result['pid']))])
        def reader(path):
            return copy.deepcopy({'caller-normal.json': result, 'shapes.json': shapes, 'identity.json': state}[path.name])
        with patch.object(checks, 'read', side_effect=reader), patch.object(checks, 'pin', return_value={}):
            return checks.caller(Path('payload'), Path('campaign'), 'normal')

    def test_complete_qualified_schedule(self):
        result = self.invoke(lambda r: None)
        self.assertEqual(result['cases'], 400)
        self.assertGreater(result['payload_differences'], 0)

    def test_missing_case_refused(self):
        with self.assertRaises(AssertionError): self.invoke(lambda r: r['records'].pop())

    def test_changed_finite_digest_refused(self):
        with self.assertRaises(AssertionError):
            self.invoke(lambda r: r['records'][0].update(digest='0' * 64))

    def test_non_nan_exception_refused(self):
        def mutate(r):
            next(x for x in r['records'] if x['nan_payload_differences'])['nan_payload_differences'][0]['candidate'] = '7f800000'
        with self.assertRaises(AssertionError): self.invoke(mutate)

    def test_guard_index_refused(self):
        def mutate(r):
            next(x for x in r['records'] if x['nan_payload_differences'])['nan_payload_differences'][0]['index'] = 0
        with self.assertRaises(AssertionError): self.invoke(mutate)


if __name__ == '__main__': unittest.main()
