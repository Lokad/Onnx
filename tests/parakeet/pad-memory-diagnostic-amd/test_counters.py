import copy
import unittest
from counters import inspect
from prepare import consumer


class Counters(unittest.TestCase):
    def fixture(self):
        return dict(ticks=50,counters=dict(begin=80,start=100,stop=150,end=180,
            allocatedBefore=100,allocatedAfter=200,gc0=1,gc1=0,gc2=0,after0=1,after1=0,after2=0,
            userBeforeUs=10,userAfterUs=20,systemBeforeUs=2,systemAfterUs=4,
            minorBefore=5,minorAfter=8,majorBefore=0,majorAfter=0,
            voluntaryBefore=0,voluntaryAfter=0,involuntaryBefore=1,involuntaryAfter=2))

    def test_valid_counter_deltas(self):
        r=inspect(self.fixture(),1000000)
        self.assertEqual((r['pad_us'],r['bracket_us'],r['minor'],r['allocated']),(50,100,3,100))
        self.assertFalse(r['collection'])

    def test_bad_bracket(self):
        x=self.fixture();x['counters']['begin']=101
        with self.assertRaises(AssertionError):inspect(x,1000000)

    def test_changed_timer(self):
        x=self.fixture();x['ticks']=51
        with self.assertRaises(AssertionError):inspect(x,1000000)

    def test_decreasing_counters(self):
        for key in ('allocatedAfter','after0','userAfterUs','systemAfterUs','minorAfter','involuntaryAfter'):
            x=self.fixture();x['counters'][key]=-1
            with self.assertRaises(AssertionError):inspect(x,1000000)

    def test_collection_detected(self):
        x=self.fixture();x['counters']['after2']=1
        self.assertTrue(inspect(x,1000000)['collection'])

    def test_original_workload_reversible(self):
        text=consumer()  # The generator checks exact restoration of all original source.
        self.assertEqual(text.count('CPUExecutionProvider.Pad(source,'),2)
        self.assertEqual(text.count('var beforeCounters = MemoryCounters.Before();'),2)
        self.assertEqual(text.count('var sample = MemoryCounters.After(beforeCounters, start, stop);'),2)


if __name__=='__main__':unittest.main()
