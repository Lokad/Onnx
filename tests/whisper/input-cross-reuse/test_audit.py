import copy,importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('reuse_audit',Path(__file__).with_name('audit.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class MetadataTests(unittest.TestCase):
    def fixture(self):
        def memory(i):return dict(allocated_total=i*1000,managed_estimate=3000,collections=[i,i//2,i//5],gc_index=i//5,last_gc_heap=2000,last_gc_fragmented=100,last_gc_committed=4000)
        return dict(context_lifecycle='one-reused-context',records=[dict(memory_before=memory(2*i),memory_after=memory(2*i+1),pool_allocated_bytes=500,pool_reused_bytes=20000) for i in range(42)])

    def check(self,result):return module.metadata(result,dict(protocol='whisper-input-cross-reused-context-v2'))

    def test_complete_metadata(self):
        result=self.fixture();summary=self.check(result)
        self.assertEqual(summary['pool_allocated_bytes'],42*500)
        self.assertEqual(summary['pool_reused_bytes'],42*20000)
        self.assertEqual(summary['last']['allocated_total'],83000)

    def test_changed_scope_and_incomplete_records(self):
        for key,value in [('context_lifecycle','fresh-contexts'),('records',self.fixture()['records'][:-1])]:
            result=self.fixture();result[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):self.check(result)

    def test_damaged_counters(self):
        for key,value in [('allocated_total',0),('managed_estimate',-1),('collections',[0,0,0]),('gc_index',0),
                          ('last_gc_fragmented',3000),('last_gc_committed',1),('collections',[1,2]),('allocated_total',1.5)]:
            result=self.fixture();result['records'][10]['memory_after'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(AssertionError):self.check(result)
        for key,value in [('pool_allocated_bytes',-1),('pool_reused_bytes',-1),('pool_allocated_bytes',1001)]:
            result=self.fixture();result['records'][10][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(AssertionError):self.check(result)

if __name__=='__main__':unittest.main()
