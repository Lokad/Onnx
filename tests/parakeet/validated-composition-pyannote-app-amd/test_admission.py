import copy,unittest
from fractions import Fraction
from admission import evaluate

def exact(value):
    f=Fraction(value);return dict(numerator=f.numerator,denominator=f.denominator)

def table(candidate=100):
    rows=[]
    for i,duration in enumerate([10,10,10,30]):
        row=dict(name=str(i),audio_seconds=duration)
        for role,value in [('selected',100),('candidate',candidate),('ort',80)]:
            row[role]=dict(exact_mean=exact(value),processes=[dict(exact_mean=exact(value)),dict(exact_mean=exact(value))])
        rows.append(row)
    return rows

class Admission(unittest.TestCase):
    def test_exact_five_percent_boundary(self):self.assertTrue(evaluate(table(105))['admitted'])
    def test_any_fixture_over_limit_rejected(self):
        for index in range(4):
            rows=table();value=Fraction(10500001,100000)
            rows[index]['candidate']=dict(exact_mean=exact(value),processes=[dict(exact_mean=exact(value))]*2)
            self.assertFalse(evaluate(rows)['admitted'])
    def test_every_engine_stability_matters(self):
        for role in ['selected','candidate','ort']:
            rows=table();row=rows[-1][role]
            row['processes']=[dict(exact_mean=exact(100)),dict(exact_mean=exact(111))]
            row['exact_mean']=exact(Fraction(211,2))
            self.assertFalse(evaluate(rows)['controls_passed'])
    def test_inconsistent_aggregate_is_rejected(self):
        rows=table();rows[0]['candidate']['exact_mean']=exact(99)
        with self.assertRaises(AssertionError):evaluate(rows)

if __name__=='__main__':unittest.main()
