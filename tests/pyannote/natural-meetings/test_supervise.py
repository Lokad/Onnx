import copy
import unittest
from supervise import check_sample


class ResourceTests(unittest.TestCase):
    def test_cpu_memory_and_time_limits_are_enforced(self):
        limits=dict(rss=8*1024**3,seconds=3600,available=1024**3)
        sample=dict(seconds=10,available=2*1024**3,members=[dict(rss=1024**3,affinity=[2])])
        check_sample(sample,limits)
        for change in [lambda s:s.update(seconds=3600),lambda s:s.update(available=1024**3-1),
                       lambda s:s['members'][0].update(rss=8*1024**3),lambda s:s['members'][0].update(affinity=[0,2])]:
            bad=copy.deepcopy(sample);change(bad)
            with self.assertRaises(AssertionError):check_sample(bad,limits)


if __name__=='__main__':unittest.main()
