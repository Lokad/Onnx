import unittest
from profiles import policy,check_runtime


class ProfileTests(unittest.TestCase):
    def test_wrong_host_runtime_affinity_and_override_are_refused(self):
        self.assertEqual(policy('windows')['rss'],20*1024**3)
        self.assertEqual(policy('amd')['rss'],14*1024**3)
        self.assertEqual(policy('amd')['preflight'],13*1024**3)
        self.assertEqual(policy('amd')['available'],1024**3)
        with self.assertRaises(ValueError):policy('unknown')
        for name,runtime in [('windows','.NET 10.0.12'),('amd','.NET 10.0.8')]:
            good=dict(runtime=runtime,affinity=4,flags=[]);check_runtime(name,good)
            for key,value in [('runtime','.NET 9.0'),('affinity',1),('flags',['DOTNET_TieredCompilation'])]:
                with self.assertRaises(AssertionError):check_runtime(name,dict(good,**{key:value}))


if __name__=='__main__':unittest.main()
