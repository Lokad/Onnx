"""Exercise the observed exit race without weakening live resource guards."""
import subprocess
import types
import unittest
from terminal_snapshot import observe


class TerminalTests(unittest.TestCase):
    def test_confirmed_exit_has_separate_receipt(self):
        child = types.SimpleNamespace(wait=lambda **kw: 0)
        value = observe(child, [], {'5': 12.0}, lambda identity: True, 2, 14*1024**3, 5*1024**3, 500)
        self.assertEqual(value['code'], 0)
        self.assertEqual(value['identities'], {'5': 12.0})

    def test_live_members_remain_in_ordinary_guard(self):
        def wait(**kw): raise AssertionError('Must not wait for live samples')
        child = types.SimpleNamespace(wait=wait)
        self.assertIsNone(observe(child, [{'rss': 20*1024**3}], {'5': 12.0}, lambda i: False, 2, 1, 1, 1))

    def test_running_root_is_not_accepted(self):
        def wait(**kw): raise subprocess.TimeoutExpired('fixture', .25)
        child = types.SimpleNamespace(wait=wait)
        with self.assertRaisesRegex(AssertionError, 'has not terminated'):
            observe(child, [], {'5': 12.0}, lambda i: True, 2, 14*1024**3, 5*1024**3, 500)

    def test_live_descendant_is_not_accepted(self):
        child = types.SimpleNamespace(wait=lambda **kw: 0)
        with self.assertRaisesRegex(AssertionError, 'descendant'):
            observe(child, [], {'5': 12.0, '6': 13.0}, lambda i: i['pid'] == 5, 2, 14*1024**3, 5*1024**3, 500)

    def test_memory_bound_still_applies_to_transition(self):
        child = types.SimpleNamespace(wait=lambda **kw: 0)
        with self.assertRaises(AssertionError):
            observe(child, [], {'5': 12.0}, lambda i: True, 2, 1, 5*1024**3, 500)


if __name__ == '__main__': unittest.main()
