from fractions import Fraction as F
import copy,json,unittest
from pathlib import Path
from gates import ORDER,evaluate
from score import fixtures,constants,check_result,PRODUCTS
ROOT=Path(__file__).resolve().parents[3]

def reports(role):
    spec=json.loads((ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922/output/result.json').read_text())
    reference=json.loads((ROOT/'artifacts/pyannote-winograd-output-blocks-numerics-amd-20260923/collected/candidate-captured-512/result.json').read_text())
    calls=fixtures(spec);expected={(r['fixture'],r['index']):r for r in reference['rows']}
    unique=constants(calls);size=sum(4*c['weights']['shape'][0]*c['weights']['shape'][1]*16 for c in unique)
    rows=[];prep=[]
    for p in range(4):
        for call in unique:
            m,c,_,_=call['weights']['shape']
            prep.append(dict(kind='preparation',role=role,pass_=p,warmup=p==0,index=call['index'],ticks=100,frequency=100,bytes=4*m*c*16,sha256='a'*64))
            prep[-1]['pass']=prep[-1].pop('pass_')
        for call in calls:
            n,c,h,w=call['input']['shape'];m=call['weights']['shape'][0]
            scratch=4*(16*c*8+16*m*8+m*h*w)
            for i in range(3):
                row=dict(kind='call',role=role,pass_=p,warmup=p==0,fixture=call['case'],index=call['index'],form=call['form'],iteration=i,iterations=3,
                    work=m*c*9*h*w,ticks=100,frequency=100,values=m*h*w,exact=True,
                    sha256=expected[(call['case'],call['index'])]['output'],
                    scratch_requested_bytes=scratch,scratch_rented_bytes=scratch,prepared_bytes=size)
                row['pass']=row.pop('pass_');rows.append(row)
    return dict(passed=True,read_only_operands=True,held_outputs=True,role=role,protocol='winograd-87-geometry-2pow31-v1',products=PRODUCTS,loaded={role:dict(sha256=PRODUCTS[role],location='/capture/runtimes/'+role+'/Lokad.Onnx.dll')},
        cases=87,calls=1044,warmups=261,measured=783,observations=rows,preparation=prep),calls,reference

class Gates(unittest.TestCase):
    def totals(self,current,candidate):return {n:dict(candidate if n.startswith('candidate') else current) for n in ORDER}
    def test_exact_weighted_boundary(self):
        r=evaluate(self.totals({0:F(9),1:F(1)},{0:F(8),1:F(1)}),{0:True,1:True})
        self.assertTrue(r['admitted']);self.assertEqual(r['rows'][0]['ratio']['seconds'],.9)
    def test_one_tick_above_boundary(self):self.assertFalse(evaluate(self.totals({0:F(10**12)},{0:F(9*10**11+1)}),{0:True})['admitted'])
    def test_form_regression(self):
        r=evaluate(self.totals({0:F(100),1:F(1)},{0:F(70),1:F(106,100)}),{0:True,1:True})
        self.assertTrue(r['gates'][0]['passed']);self.assertFalse(r['admitted'])
    def test_current_control(self):
        t=self.totals({0:F(100)},{0:F(50)});t['current-b'][0]=F(111)
        self.assertFalse(evaluate(t,{0:True})['admitted'])
    def test_form_control(self):
        t=self.totals({0:F(100),1:F(1)},{0:F(50),1:F(1,2)});t['current-b'][1]=F(121,100)
        r=evaluate(t,{0:True,1:True});self.assertTrue(r['controls'][0]['passed']);self.assertFalse(r['admitted'])
    def test_separation_equality(self):
        t=self.totals({0:F(100)},{0:F(80)});t['current-b'][0]=F(110);t['candidate-b'][0]=F(100)
        self.assertFalse(evaluate(t,{0:True})['process_separation']['passed'])
    def test_nonpositive(self):
        with self.assertRaises(AssertionError):evaluate(self.totals({0:F(0)},{0:F(1)}),{0:True})

class Clocks(unittest.TestCase):
    def test_complete_weighting(self):
        for role in ['current','candidate']:
            data=reports(role);totals,_,_=check_result(*data)
            self.assertEqual(sum(totals.values()),87)
    def test_warmup_not_scored(self):
        data=reports('candidate');data[0]['observations'][0]['ticks']=10**12
        self.assertEqual(sum(check_result(*data)[0].values()),87)
    def test_one_measured_tick_exact(self):
        data=reports('current');data[0]['observations'][261]['ticks']+=1
        self.assertEqual(sum(check_result(*data)[0].values()),F(87)+F(1,900))
    def test_missing_duplicate_wrong_work(self):
        for change in ['missing','duplicate','work','scratch','preparation','hash']:
            data=reports('candidate');value=data[0]
            if change=='missing':value['observations'].pop()
            elif change=='duplicate':value['observations'][1]=copy.deepcopy(value['observations'][0])
            elif change=='work':value['observations'][0]['work']+=1
            elif change=='scratch':value['observations'][0]['scratch_requested_bytes']=0
            elif change=='preparation':value['preparation'].pop()
            else:value['observations'][0]['sha256']='0'*64
            with self.assertRaises(AssertionError):check_result(*data)
    def test_old_direct_output_cannot_be_substituted(self):
        data=reports('candidate');data[0]['observations'][0]['sha256']=data[2]['rows'][0]['selectedOutput']
        with self.assertRaises(AssertionError):check_result(*data)
    def test_zero_tick(self):
        data=reports('current');data[0]['observations'][0]['ticks']=0
        with self.assertRaises(AssertionError):check_result(*data)
    def test_wrong_loaded_product(self):
        for role in ['current','candidate']:
            data=reports(role);data[0]['loaded'][role]['sha256']=PRODUCTS['candidate' if role=='current' else 'current']
            with self.assertRaises(AssertionError):check_result(*data)
    def test_unloaded_role(self):
        data=reports('current');data[0]['loaded']={}
        with self.assertRaises(AssertionError):check_result(*data)
    def test_old_direct_preparation_is_rejected(self):
        data=reports('current');data[0]['preparation'][0]['bytes']=data[0]['preparation'][0]['bytes']//16*9
        with self.assertRaises(AssertionError):check_result(*data)

if __name__=='__main__':unittest.main()
