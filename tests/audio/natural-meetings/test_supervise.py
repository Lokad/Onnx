import copy
import unittest
from common import limits
from supervise import check_sample


class ResourceTests(unittest.TestCase):
    def test_limits_refuse_bad_samples_for_each_host(self):
        for host in ['windows','amd']:
            policy=limits(host)
            original=dict(seconds=1,available=policy['available'],members=[dict(pid=10,birth=1,rss=100,affinity=[2])])
            check_sample(original,policy)
            mutations=[lambda s:s.update(seconds=policy['seconds']),lambda s:s.update(available=policy['available']-1),
                lambda s:s['members'][0].update(rss=policy['rss']),lambda s:s['members'][0].update(affinity=[0])]
            for mutation in mutations:
                bad=copy.deepcopy(original);mutation(bad)
                with self.assertRaises(AssertionError):check_sample(bad,policy)


if __name__=='__main__':unittest.main()
