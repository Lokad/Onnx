"""Exercise controller handoff and terminal failures without launching processes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import finish


class FinishTests(unittest.TestCase):
    def scenario(self, remote_code=0, stage_code=0, collection_code=0, audit_code=0, pending=False):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); (base/'prepared.json').write_text('{}')
            e5 = base/'e5.json'
            e5.write_text(json.dumps(dict(complete=not pending, supervisor=dict(pid=2, birth=10 if pending else 9), stages=[])))
            actions = []; waits = []
            class Process:
                def __init__(self, pid=1): self.pid = pid
                def create_time(self): return 10
                def cpu_affinity(self, value): pass
            def spawn(command, **kwargs):
                self.assertEqual(command[1:4], ['-X', 'utf8', '-B'])
                name = Path(command[4]).name; action = command[5] if len(command) > 5 else ''
                actions.append((name, action))
                code = {'stage': stage_code, 'collect': collection_code}.get(action, audit_code if name == 'audit_results.py' else 0)
                return types.SimpleNamespace(pid=3, wait=lambda **k: code, poll=lambda: code)
            def sleep(seconds):
                waits.append(seconds); e5.write_text(json.dumps(dict(complete=True, supervisor=dict(pid=2, birth=9), stages=[])))
            with contextlib.ExitStack() as stack:
                for key, value in dict(BASE=base, ROOT=base, TOOLS=base, SITE=base, E5_CONTROL=e5).items():
                    stack.enter_context(patch.object(finish, key, value))
                stack.enter_context(patch.dict(finish.sys.modules, psutil=types.SimpleNamespace(Process=Process)))
                stack.enter_context(patch.object(finish, 'checked_local'))
                stack.enter_context(patch.object(finish, 'local_e5_terminal'))
                stack.enter_context(patch.object(finish, 'observe', return_value=dict(supervisor_live=False, live=[], complete=True, code=remote_code)))
                stack.enter_context(patch.object(finish.subprocess, 'Popen', spawn))
                stack.enter_context(patch.object(finish.time, 'sleep', sleep))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                code = finish.main()
            state = json.loads((base/'controller/state.json').read_text())
            self.assertTrue(state['complete']); self.assertEqual(code, state['code'])
            return code, actions, waits

    def test_wait_for_existing_owner_then_launch_once(self):
        code, actions, waits = self.scenario(pending=True)
        self.assertEqual(code, 0); self.assertEqual(waits, [30])
        self.assertEqual(actions, [('transport.py', 'stage'), ('transport.py', 'launch'),
                                   ('transport.py', 'collect'), ('audit_results.py', '')])

    def test_remote_failure_is_collected_and_audited_without_retry(self):
        code, actions, _ = self.scenario(remote_code=1, audit_code=1)
        self.assertEqual(code, 1)
        self.assertEqual(actions[-2:], [('transport.py', 'collect'), ('audit_results.py', '')])
        self.assertEqual(actions.count(('transport.py', 'launch')), 1)

    def test_failed_stage_prevents_launch(self):
        code, actions, _ = self.scenario(stage_code=1)
        self.assertEqual(code, 1); self.assertEqual(actions, [('transport.py', 'stage')])

    def test_failed_collection_prevents_audit(self):
        code, actions, _ = self.scenario(collection_code=1)
        self.assertEqual(code, 1); self.assertNotIn(('audit_results.py', ''), actions)


if __name__ == '__main__':
    unittest.main()
