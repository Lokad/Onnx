"""Reject incomplete native-node coverage, invalid arrays and resource overruns."""
import copy
import unittest
import numpy as np
from protocol import LIMITS, check_sample, compare, profile_events


class ProtocolTests(unittest.TestCase):
    def test_native_coverage(self):
        census = dict(nodes=[dict(name='conv',scope='main',domain='com.microsoft.nchwc',op='Conv')])
        event = dict(cat='Node',name='conv_kernel_time',dur=1,args=dict(op_name='Conv',provider='CPUExecutionProvider'))
        events = [copy.deepcopy(event) for _ in range(6)]+[dict(name='model_run',cat='Session') for _ in range(6)]
        self.assertEqual(len(profile_events(events,census)['kernels']),6)
        with self.assertRaises(AssertionError): profile_events(events[1:],census)
        events[0]['args']['provider'] = 'Other'
        with self.assertRaises(AssertionError): profile_events(events,census)

    def test_nested_branch_and_unknown_node(self):
        census = dict(nodes=[dict(name='if',scope='main',domain='',op='If'),
                            dict(name='untaken',scope='main/0/else_branch',domain='',op='Identity')])
        events = [dict(cat='Node',name='if_kernel_time',dur=1,args=dict(op_name='If',provider='CPUExecutionProvider')) for _ in range(6)]
        events += [dict(name='model_run',cat='Session') for _ in range(6)]
        profile_events(events,census)
        events[0]['name'] = 'missing_kernel_time'
        with self.assertRaises(AssertionError): profile_events(events,census)

    def test_numerical_bound_and_nonfinite(self):
        x = np.array([1.,-2.],dtype=np.float32)
        self.assertTrue(compare(x,x)['identical'])
        self.assertEqual(compare(x+np.float32(.001),x)['failed_values'],2)
        with self.assertRaises(AssertionError): compare(x,np.array([float('nan'),1],dtype=np.float32))

    def test_resource_boundaries_and_affinity(self):
        sample = dict(seconds=1,rss=100,available=LIMITS['available'],tmpfs=LIMITS['tmpfs'],output=1,artifacts=1,
            members=[dict(rss=100,affinity=[2],threads=[dict(affinity=[2])])])
        check_sample(sample)
        for field,value in [('seconds',900),('available',LIMITS['available']-1),('tmpfs',LIMITS['tmpfs']-1),('output',LIMITS['output']+1),('artifacts',LIMITS['artifacts']+1)]:
            broken = copy.deepcopy(sample); broken[field] = value
            with self.assertRaises(AssertionError): check_sample(broken)
        sample['members'][0]['threads'][0]['affinity'] = [0]
        with self.assertRaises(AssertionError): check_sample(sample)


if __name__ == '__main__': unittest.main()
