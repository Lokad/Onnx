import copy, unittest
from shared import *
from generate import generate


class Tests(unittest.TestCase):
    def test_exact_rounding_groups(self):
        source=PRODUCT.read_bytes()
        for variant in VARIANTS:
            text=generate(source,variant)
            self.assertEqual('new double[WindowSize]' in text,variant in ['Frame','All'])
            self.assertEqual('new double[FourierSize / 2 + 1]' in text,variant in ['Spectrum','All'])
            self.assertEqual('var output = new double[frames * MelBins]' in text,variant in ['Output','All'])
            self.assertEqual('MathF.Log(Math.Max(Epsilon, (float)energy))' in text,variant not in ['Output','All'])
            self.assertIn('static readonly float[] Window = Tables.Window.ToArray();',text)
            for stage in ['Window','Fourier','Power','Energy','Log','Features']:self.assertEqual(text.count('Capture.'+stage+'('),1)
            self.assertIn('static void Fourier(Span<Complex> data)',text)

    def test_source_drift_refused(self):
        for source in [PRODUCT.read_bytes()+b' ',PRODUCT.read_bytes().replace(b'.97f',b'.98f')]:
            with self.assertRaises(AssertionError):generate(source,'All')
        with self.assertRaises(AssertionError):generate(PRODUCT.read_bytes(),'unknown')

    def test_complete_arrays_and_aggregation(self):
        target=np.array([[1.,2.,3.]],np.float64);actual=np.array([[1.,2.0001,3.01]],np.float64)
        rows=[comparison(actual,target),comparison(target,target)]
        result=aggregate(rows);self.assertEqual(result['values'],6);self.assertEqual(result['failed'],1)
        self.assertAlmostEqual(result['squared_error'],.00010001)
        for value in [np.zeros((1,2)),np.zeros((1,3),np.float32),np.array([[1.,np.nan,3.]])]:
            with self.assertRaises(AssertionError):check_array(value,[1,3])
        for count in [399,480001,400.0]:
            with self.assertRaises(AssertionError):shapes(count)
        self.assertEqual(shapes(559)['features'],[1,1,80]);self.assertEqual(shapes(560)['features'],[1,2,80])

    def test_resource_refusals(self):
        state=dict(complete=True,code=0,limits=LIMITS,supervisor=dict(pid=1,birth=1),runs=[]);samples={}
        for i,name in enumerate(['build','managed','numpy','torch']):
            pid=i+2;member=dict(pid=pid,birth=pid,rss=1000,affinity=[0])
            state['runs'].append(dict(name=name,complete=True,code=0,seconds=1,started=pid,ended=pid+1,
                preflight_available=LIMITS['preflight'],preflight_disk=LIMITS['disk'],members={str(pid):pid},child=dict(pid=pid,birth=pid),samples=2))
            samples[name]=[dict(seconds=t,members=[member],available=LIMITS['available']) for t in [.1,.6]]
        self.assertEqual(len(resource_checks(state,samples)[0]),5)
        for key,value in [('code',1),('complete',False),('limits',{})]:
            bad=copy.deepcopy(state);bad[key]=value
            with self.assertRaises(AssertionError):resource_checks(bad,samples)
        for key,value in [('seconds',2),('available',0),('members',[dict(member,affinity=[2])]),('members',[dict(member,rss=LIMITS['rss'])])]:
            bad=copy.deepcopy(samples);bad['build'][0][key]=value
            with self.assertRaises(AssertionError):resource_checks(state,bad)


if __name__=='__main__':unittest.main()
