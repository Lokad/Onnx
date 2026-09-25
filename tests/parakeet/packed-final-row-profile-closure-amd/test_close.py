import copy
import unittest
from close import resume, read, verify_collision


class CollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lines=(resume.BASE/'observations.jsonl').read_text(encoding='utf8').splitlines()
        import json
        cls.actual=json.loads(lines[-1])['state']
        assert cls.actual['complete'] and cls.actual['code']==1

    def pair(self):
        state=copy.deepcopy(self.actual)
        checkpoint=dict(state,complete=False,code=None)
        del checkpoint['error'];del checkpoint['ended']
        return state,checkpoint

    def test_exact_failure(self):
        verify_collision(*self.pair())

    def test_other_failure_rejected(self):
        state,checkpoint=self.pair();state['error']='other failure'
        with self.assertRaises(AssertionError):verify_collision(state,checkpoint)

    def test_failed_process_rejected(self):
        state,checkpoint=self.pair();state['runs'][0]['code']=1
        with self.assertRaises(AssertionError):verify_collision(state,checkpoint)

    def test_different_checkpoint_rejected(self):
        state,checkpoint=self.pair();checkpoint['started']+=1
        with self.assertRaises(AssertionError):verify_collision(state,checkpoint)

    def test_success_relabel_rejected(self):
        state,checkpoint=self.pair();state['code']=0
        with self.assertRaises(AssertionError):verify_collision(state,checkpoint)


if __name__=='__main__':unittest.main()
