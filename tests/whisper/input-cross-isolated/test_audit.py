from pathlib import Path
import copy,importlib.util,unittest
from protocol import schedule,coverage
spec=importlib.util.spec_from_file_location('isolated_audit',Path(__file__).with_name('audit.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ProtocolTests(unittest.TestCase):
    def fixture(self):
        result=dict(requests=[dict(name=f'case-{i}') for i in list(range(20))+[0]])
        result['schedule']=schedule(result);return result

    def test_full_scope_and_order(self):
        spec=self.fixture();jobs=spec['schedule'];coverage(jobs,spec)
        self.assertEqual([j['engine'] for j in jobs],['managed']*21+['native']*21)
        self.assertEqual([j['request'] for j in jobs],list(range(21))*2)
        for damaged in [jobs[:-1],jobs[::-1],jobs[:20]+jobs[21:]+[jobs[20]],jobs[:-1]+[jobs[0]]]:
            with self.assertRaises(AssertionError):coverage(damaged,spec)

    def test_worker_identity_and_baseline_refusals(self):
        job=self.fixture()['schedule'][3]
        result=dict(complete=True,engine='managed',request_index=3,manifest_sha256='digest',flags={},
                    records=[dict(request=3,name=job['name'],kind=k,baseline_matches=True if k=='MM' else None) for k in ['MM','MN']])
        module.header(result,job,'digest')
        for key,value in [('complete',False),('engine','native'),('request_index',4),('manifest_sha256','wrong'),('flags',{'flag':'1'}),('records',[])]:
            damaged=copy.deepcopy(result);damaged[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):module.header(damaged,job,'digest')
        for key,value in [('request',2),('name','other'),('kind','MN'),('baseline_matches',False)]:
            damaged=copy.deepcopy(result);damaged['records'][0][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):module.header(damaged,job,'digest')

if __name__=='__main__':unittest.main()
