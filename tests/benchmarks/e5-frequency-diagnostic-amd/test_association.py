import copy
import unittest
from association import associate
from counter import EVENTS
from protocol import check_sample


def fixture():
    clocks=[dict(frequency=10**9,start=(i+20)*10**9,end=(i+21)*10**9,ticks=10**9) for i in range(780)]
    rows=[dict(elapsed_ns=i*10**9,events={e:dict(count='200' if e==EVENTS[0] else '100',running_percent='100') for e in EVENTS}) for i in range(1,802)]
    return clocks,rows,dict(lower_ns=0,upper_ns=20_000_000)


class AssociationTests(unittest.TestCase):
    def test_all_intervals_retained_and_no_boundary_interpolation(self):
        clocks,rows,bounds=fixture();result=associate(clocks,rows,bounds)
        self.assertEqual(len(rows),len(result['intervals']))
        for block in result['blocks']:
            self.assertEqual(block['aperf_mperf'],2)
            self.assertEqual(block['covered_ns'],59*10**9)
            self.assertEqual(block['wall_ms'],1000)
        self.assertIsNone(result['intervals'][79]['block_first'])
        self.assertEqual(result['intervals'][80]['block_first'],60)

    def test_reject_insufficient_coverage(self):
        clocks,rows,bounds=fixture();bounds['upper_ns']=11*10**9
        with self.assertRaises(AssertionError):associate(clocks,rows,bounds)

    def test_reject_unavailable_inside_block(self):
        clocks,rows,bounds=fixture();rows[30]['events'][EVENTS[0]]['count']=None
        with self.assertRaises(AssertionError):associate(clocks,rows,bounds)

    def test_reject_multiplexing_inside_block(self):
        clocks,rows,bounds=fixture();rows[30]['events'][EVENTS[0]]['running_percent']='99'
        with self.assertRaises(AssertionError):associate(clocks,rows,bounds)

    def test_keep_unavailable_boundary_interval(self):
        clocks,rows,bounds=fixture();rows[79]['events'][EVENTS[0]]['count']=None
        result=associate(clocks,rows,bounds)
        self.assertIsNone(result['intervals'][79]['events'][EVENTS[0]]['count'])

    def test_frequency_affinity_exception_does_not_apply_to_worker(self):
        member=dict(role='frequency',rss=100,expected_affinity=[0],affinity=[2],threads=[dict(affinity=[2])])
        row=dict(seconds=1,members=[member],rss=100,available=2*1024**3,tmpfs=2*1024**3,output=0,artifacts=0)
        check_sample(row)
        member['role']='worker'
        with self.assertRaises(AssertionError):check_sample(row)
        member['role']='frequency';member['threads'][0]['affinity']=[0,2]
        with self.assertRaises(AssertionError):check_sample(row)


if __name__=='__main__':unittest.main()
