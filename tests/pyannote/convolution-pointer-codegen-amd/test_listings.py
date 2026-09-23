"""Check exact repair against independently retained bodies and reject truncation."""
import json
from pathlib import Path
import unittest
from listings import listings,MARKER

ROOT=Path(__file__).resolve().parents[3]
class Text:
    def __init__(self,text):self.text=text
    def read_text(self,encoding):return self.text

class Listings(unittest.TestCase):
    def test_retained_independent_bodies(self):
        expected=json.loads((ROOT/'tests/pyannote/convolution-channel-results/normalized-listings-20260923.json').read_text())
        for role in ['production','candidate']:
            path=ROOT/'artifacts/pyannote-convolution-channel-codegen-amd-20260923/collected/logs'/(role+'.stdout')
            actual=listings(path)
            self.assertEqual(len(actual),len(expected[role]))
            for a,e in zip(actual,expected[role],strict=True):
                self.assertEqual(a['body'],e['body'])
                self.assertEqual(a['method'],e['method']);self.assertEqual(a['code_bytes'],e['code_bytes'])
                self.assertTrue(a['complete_body'])
                self.assertEqual(a['complete_uninterleaved'],not a['managed_stdout_repairs'])
                self.assertEqual(a['raw_body'].replace(MARKER,''),a['body'])
            self.assertEqual(sum(len(a['managed_stdout_repairs']) for a in actual),1)

    def source(self):
        return (ROOT/'artifacts/pyannote-convolution-channel-codegen-amd-20260923/collected/logs/candidate.stdout').read_text()

    def test_missing_footer_fails(self):
        rows=listings(Text(self.source().rsplit('; Total bytes of code ',1)[0]))
        self.assertFalse(rows[-1]['complete_body'])

    def test_duplicate_block_fails(self):
        rows=listings(Text(self.source()+'\nG_M000_IG01: ;; duplicate\n'))
        self.assertFalse(rows[-1]['complete_body'])

    def test_unknown_message_is_not_removed(self):
        text=self.source().replace(MARKER,MARKER.replace('432','431'))
        row=listings(Text(text))[-1]
        self.assertFalse(row['complete_body']);self.assertEqual(row['managed_stdout_repairs'],[])

    def test_missing_block_fails(self):
        text=self.source();start=text.rfind('G_M000_IG110:');end=text.index('G_M000_IG111:',start)
        row=listings(Text(text[:start]+text[end:]))[-1]
        self.assertFalse(row['complete_body'])

if __name__=='__main__':unittest.main()
