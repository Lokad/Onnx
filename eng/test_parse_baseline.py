"""Historical baseline CLI selection contracts; synthetic logs only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import parse_baseline as baseline


def case(name, status='ok'):
    return (f'case-status {name}={status}\n{name} [lokad vs ort]: synthetic\n'
            'raw lok=[1,2] raw ctx=[1,2] raw ort=[2,3]\n')


class ParseBaselineTests(unittest.TestCase):
    def invoke(self, logs, requested=None):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, content in enumerate(logs):
                path = Path(directory) / f'rep-{i}.log'
                path.write_text(content, encoding='utf-8')
                paths.append(str(path))
            command = [sys.executable, str(Path(baseline.__file__).resolve())]
            if requested is not None:
                command += ['--cases', requested]
            return subprocess.run(command + paths, capture_output=True, text=True)

    def rows(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [row['case'] for row in json.loads(result.stdout[result.stdout.index('{'):])['table']]

    def test_unknown_and_empty_requests_fail(self):
        for requested in ['typo-case', 'gemma2-128tok', '', 'e5-8tok,typo-case']:
            with self.subTest(requested=requested):
                result = self.invoke([''], requested)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('unknown', result.stderr.lower())

    def test_missing_explicit_primary_and_extended_fail(self):
        for requested in ['e5-8tok', 'e5-128tok', 'e5-8tok,e5-128tok']:
            with self.subTest(requested=requested):
                result = self.invoke([''], requested)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('case missing or not ok', result.stderr)

    def test_explicit_case_is_required_in_every_rep(self):
        result = self.invoke([case('e5-128tok'), ''], 'e5-128tok')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('rep-1.log', result.stderr)

    def test_failed_requested_case_is_rejected(self):
        result = self.invoke([case('e5-128tok', 'FAILED')], 'e5-128tok')
        self.assertNotEqual(result.returncode, 0)

    def test_valid_extended_only_request_does_not_require_primary(self):
        self.assertEqual(self.rows(self.invoke([case('e5-128tok')] * 2, 'e5-128tok')), ['e5-128tok'])

    def test_mixed_and_duplicate_requests_keep_canonical_order(self):
        content = case('e5-128tok') + case('e5-8tok')
        self.assertEqual(self.rows(self.invoke([content], 'e5-128tok,e5-8tok,e5-128tok')), ['e5-8tok', 'e5-128tok'])

    def test_default_primary_required_and_extended_optional(self):
        content = ''.join(case(name) for name in baseline.ORDER)
        self.assertEqual(self.rows(self.invoke([content + case('e5-128tok'), content])), baseline.ORDER)
        self.assertEqual(self.rows(self.invoke([content + case('e5-128tok')] * 2)), baseline.ORDER + ['e5-128tok'])
        self.assertNotEqual(self.invoke([content.replace(case('e5-8tok'), '')]).returncode, 0)


if __name__ == '__main__':
    unittest.main()
