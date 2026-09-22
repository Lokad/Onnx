import tempfile
from pathlib import Path
import unittest
from transport import PREPARED
from resume_prefix import restore_prefix, verify_prefix, PREFIX


class ResumeTests(unittest.TestCase):
    def test_exact_successful_prefix_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as name:
            campaign = Path(name)
            value = restore_prefix(PREPARED / 'payload', campaign, lambda identity: True)
            self.assertEqual([r['name'] for r in value['runs']], PREFIX)
            self.assertEqual(len(value['runs']), 14)
            self.assertFalse((campaign / 'portable-pyannote').exists())
            self.assertFalse((campaign / 'identity.json').exists())
            self.assertEqual(verify_prefix(PREPARED / 'payload', campaign), value)
            with (campaign / 'operator-gate.json').open('ab') as stream:
                stream.write(b' ')
            with self.assertRaises(AssertionError):
                verify_prefix(PREPARED / 'payload', campaign)

    def test_live_previous_worker_refuses_before_copy(self):
        with tempfile.TemporaryDirectory() as name:
            campaign = Path(name)
            with self.assertRaises(AssertionError):
                restore_prefix(PREPARED / 'payload', campaign, lambda identity: False)
            self.assertEqual(list(campaign.iterdir()), [])
