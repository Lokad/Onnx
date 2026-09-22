"""Check each real closed-input schema before any preparation writes."""
import unittest
from prepare import PREREQUISITES
from candidate_protocol import pin, read

class PrerequisiteTests(unittest.TestCase):
    def test_all_exact_closed_inputs_have_the_named_terminal_schema(self):
        self.assertEqual(len(PREREQUISITES), 5)
        for folder, sha, key in PREREQUISITES:
            path = folder / 'closed.json'
            self.assertEqual(pin(path)['sha256'], sha)
            value = read(path); self.assertTrue(value['passed'])
            self.assertTrue(value[key])
            self.assertTrue(all(set(i) == {'pid', 'birth'} for i in value[key]))
