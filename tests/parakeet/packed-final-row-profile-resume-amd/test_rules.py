import ast
import copy
import unittest
from checks import ORIGINAL,PROFILE,read,original_checker
from rules import verify_refusal
from worker import adapted_capture


class ResumeOnlyUnstarted(unittest.TestCase):
    def setUp(self):
        self.state=read(PROFILE/'capture-collected/capture-state.json')
        self.spec=read(PROFILE/'bundle/spec.json')

    def test_actual_refusal_allows_only_missing_modes(self):
        result=verify_refusal(self.state,self.spec)
        self.assertEqual(result['completed'],['control'])
        self.assertEqual(result['unstarted'],['phase','wall'])
        self.assertEqual(result['memory_shortfall'],17776640)

    def test_started_phase_cannot_be_repeated(self):
        self.state['runs'][1]['members']={'123':456.0}
        with self.assertRaises(AssertionError):verify_refusal(self.state,self.spec)

    def test_missing_completed_control_is_rejected(self):
        self.state['runs'][0]['complete']=False
        with self.assertRaises(AssertionError):verify_refusal(self.state,self.spec)

    def test_execution_failure_cannot_be_called_preflight(self):
        self.state['runs'][1]['seconds']=1
        with self.assertRaises(AssertionError):verify_refusal(self.state,self.spec)

    def test_different_failure_is_rejected(self):
        self.state['runs'][1]['preflight']['available']=self.spec['capture_limits']['available_before']
        with self.assertRaises(AssertionError):verify_refusal(self.state,self.spec)

    def test_per_process_audit_body_is_exact(self):
        checker,attribute=original_checker()
        self.assertTrue(callable(checker) and callable(attribute))

    def test_capture_changes_only_modes_and_prelaunch_wait(self):
        source=(ORIGINAL/'remote.py').read_text(encoding='utf8')
        original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='capture')
        expected=copy.deepcopy(original)
        loop=next(n for n in expected.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='mode')
        loop.iter=ast.parse("['phase','wall']",mode='eval').body
        loop.body.insert(0,ast.parse('wait_for_headroom(state,mode,spec)').body[0])
        self.assertEqual(ast.dump(expected),ast.dump(ast.parse(adapted_capture(source)).body[0]))

    def test_unexpected_original_loop_is_rejected(self):
        source=(ORIGINAL/'remote.py').read_text(encoding='utf8').replace("['control','phase','wall']","['phase','wall']")
        with self.assertRaises(AssertionError):adapted_capture(source)


if __name__=='__main__':unittest.main()
