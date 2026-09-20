"""Use actual first-cohort Linux evidence to isolate the old clock-domain error."""
from pathlib import Path
import copy,json,sys,unittest
import audit_birth_clock as fixed

BASE=Path(__file__).resolve().parents[3]/'artifacts/e5-interleaved-processes-v3-20260920/birth-clock-review'

class ClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value=json.loads((BASE/'remote-evidence.json').read_text());state=value['original_state'];run=state['runs'][0]
        cls.state=state|dict(complete=True,code=0,ended=run['ended'],runs=[run])
        cls.records={run['job']['name']:value['first_cohort_samples']};cls.jobs=[run['job']]
        assert min(w['birth']-run['started'] for w in run['workers'].values())<-.05

    def test_original_refuses_valid_cross_domain_timestamps(self):
        with self.assertRaises(AssertionError):fixed.frozen_telemetry(self.state,self.records,self.jobs)

    def test_corrected_accepts_complete_actual_cohort(self):
        value=fixed.telemetry(self.state,self.records,self.jobs)
        self.assertEqual(5,len(value['births']));self.assertEqual(1,len(value['cohorts']))

    def test_older_birth_and_wall_order_still_refused(self):
        for change in [lambda v:v['runs'][0]['workers']['A'].update(birth=v['supervisor']['birth']-1),
                       lambda v:v['runs'][0]['workers']['A'].update(started=v['runs'][0]['started']-1)]:
            damaged=copy.deepcopy(self.state);change(damaged)
            with self.assertRaises(AssertionError):fixed.telemetry(damaged,self.records,self.jobs)

    def test_inactive_running_state_still_refused(self):
        damaged=copy.deepcopy(self.records)
        row=next(r for r in damaged[self.jobs[0]['name']] if any(m['role']!=r['active'] for m in r['members']))
        next(m for m in row['members'] if m['role']!=row['active'])['status']='running'
        with self.assertRaises(AssertionError):fixed.telemetry(self.state,damaged,self.jobs)

if __name__=='__main__':unittest.main()
