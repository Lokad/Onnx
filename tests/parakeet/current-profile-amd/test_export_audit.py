"""Mutation checks for export resource accounting and owner constraints."""
from copy import deepcopy
import unittest
from export_audit import check_sample


class ExportResources(unittest.TestCase):
    def test_rejects_resource_and_owner_mutations(self):
        identity=dict(pid=42,birth=123.5)
        sample=dict(seconds=1,rss=4096,available=2*1024**3,disk=2*1024**3,output_bytes=100,artifacts=200,
            members=[dict(**identity,rss=4096,affinity=[0],threads=[dict(id=42,affinity=[0])])])
        check_sample(sample,identity)
        mutations=[lambda s:s.update(seconds=900),lambda s:s.update(rss=8*1024**3),
            lambda s:s.update(available=0),lambda s:s.update(disk=0),
            lambda s:s.update(output_bytes=1024**3+1),lambda s:s.update(artifacts=2*1024**3+1),
            lambda s:s.update(rss=4095),lambda s:s['members'][0].update(pid=43),
            lambda s:s['members'][0].update(birth=124),lambda s:s['members'][0].update(affinity=[2]),
            lambda s:s['members'][0].update(threads=[]),lambda s:s['members'][0]['threads'][0].update(affinity=[2])]
        for change in mutations:
            damaged=deepcopy(sample);change(damaged)
            with self.assertRaises(AssertionError):check_sample(damaged,identity)


if __name__=='__main__':unittest.main()
