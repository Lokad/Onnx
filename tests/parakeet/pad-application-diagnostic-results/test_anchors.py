import unittest
from review_anchors import ROOT, reconcile
import sys
sys.path.insert(0,str(ROOT/'tests/parakeet/pad-application-diagnostic-amd'))
from test_diagnostic import fixture
from collections import Counter


def missing_named():
    events,summary,anchors,records,pid=fixture()
    events=[e for e in events if e['provider']!='Lokad-Pyannote-Diagnostic']
    for i,event in enumerate(events):event['index']=i
    summary['recorded']=len(events)
    summary['allCounts']=dict(Counter(e['provider']+':'+e['name'] for e in events))
    return events,summary,anchors,records,pid


class AnchorTests(unittest.TestCase):
    def test_original_failure_stays_explicit(self):
        result=reconcile(*missing_named())
        self.assertFalse(result['original_marker_gate_passed'])
        self.assertEqual(result['paired_request_anchors'],160)

    def test_missing_anchor_rejected(self):
        args=missing_named();args[2]['anchors'].pop()
        with self.assertRaises(AssertionError):reconcile(*args)

    def test_wrong_request_boundary_rejected(self):
        args=missing_named();args[3][20]['start_ticks']=args[3][19]['start_ticks']
        with self.assertRaises(AssertionError):reconcile(*args)

    def test_wrong_native_thread_rejected(self):
        args=missing_named();args[3][0]['thread_id']+=1
        with self.assertRaises(AssertionError):reconcile(*args)


if __name__=='__main__':unittest.main()
