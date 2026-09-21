"""Mathematical fixtures independent of model timings."""
from fractions import Fraction
import unittest
from analyze import components

def independent(table):
    mean=Fraction(sum(map(sum,table)),144)
    rows=[Fraction(sum(row),48) for row in table]
    columns=[Fraction(sum(table[r][c] for r in range(3)),3) for c in range(48)]
    return dict(total=sum((x-mean)**2 for row in table for x in row),
        process=48*sum((r-mean)**2 for r in rows),cycle=3*sum((c-mean)**2 for c in columns),
        residual=sum((table[r][c]-rows[r]-columns[c]+mean)**2 for r in range(3) for c in range(48)))

class ComponentsTests(unittest.TestCase):
    def test_exact_constant_and_pure_components(self):
        fixtures=[([[10]*48 for _ in range(3)],None),([[10+r]*48 for r in range(3)],'process'),
            ([[10+c for c in range(48)] for _ in range(3)],'cycle'),
            ([[10+([1,-1,0][r])*(1 if c%2 else -1) for c in range(48)] for r in range(3)],'residual')]
        for table,kind in fixtures:
            actual=components(table)
            for key,value in independent(table).items():self.assertEqual(Fraction(actual[key],144**2),value)
            for key in ['process','cycle','residual']:self.assertEqual(actual['shares'][key],float(key==kind))

    def test_mixed_components_and_large_timer_values(self):
        table=[[10**15+r*71+c*3+((r+c)%5)*11 for c in range(48)] for r in range(3)]
        actual=components(table)
        for key,value in independent(table).items():self.assertEqual(Fraction(actual[key],144**2),value)

    def test_malformed_tables_are_rejected(self):
        for table in [[],[[1]*48]*2,[[1]*47]*3,[[1.0]*48]*3,[[-1]*48]*3]:
            with self.assertRaises(AssertionError):components(table)

if __name__=='__main__':unittest.main()
