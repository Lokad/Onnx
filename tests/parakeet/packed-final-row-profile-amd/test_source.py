"""Reject observations that change the product or omit its packed-weight setup."""
import unittest
from prepare_source import SOURCE,TRANSCRIBER,PREPARE,HOOK,instrument,verify_hook


class ObservationSource(unittest.TestCase):
    def setUp(self):
        self.original=(SOURCE/'source'/TRANSCRIBER).read_bytes()
        self.observed=instrument(self.original)

    def test_original_product_recovered_exactly(self):
        self.assertTrue(verify_hook(self.original,self.observed)['owned_weight_preparation_preserved'])

    def test_legacy_observer_without_packing_is_rejected(self):
        with self.assertRaises(AssertionError):
            verify_hook(self.original,self.observed.replace(PREPARE,b'',1))

    def test_unrelated_transcription_change_is_rejected(self):
        with self.assertRaises(AssertionError):verify_hook(self.original,self.observed+b'// unrelated change\n')

    def test_hook_outside_original_entry_is_rejected(self):
        newline=b'\r\n' if b'\r\n' in self.original else b'\n'
        hook=HOOK.replace(b'\n',newline)
        moved=self.observed.replace(hook,b'',1)+hook
        with self.assertRaises(AssertionError):verify_hook(self.original,moved)


if __name__=='__main__':unittest.main()
