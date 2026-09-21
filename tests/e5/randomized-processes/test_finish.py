"""Exercise controller launch decisions without subprocesses or VM access."""
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
    def scenario(self, aa_pass, whisper_code=0, collect_code=0):
        with tempfile.TemporaryDirectory(prefix='e5-controller-') as temporary:
            root = Path(temporary)
            base, monitor, whisper, control = [root/name for name in ['base', 'whisper-monitor', 'whisper', 'control']]
            for p in [base, monitor, whisper]:
                p.mkdir()
            def put(path, value):
                path.write_text(json.dumps(value), encoding='utf8')
            put(base/'prepared.json', dict(passed=True))
            put(monitor/'state.json', dict(complete=True, code=whisper_code, supervisor=dict(pid=2, birth=1)))
            put(whisper/'final-verification.json', dict(passed=True))
            for name in ['tests/e5/process-uncertainty/estimator.py', 'tests/e5/process-uncertainty/protocol.py',
                         'tests/e5/process-uncertainty/worker_audit.py', 'eng/campaign_processes.py']:
                p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# synthetic fixture')
            (root/'tests/e5/randomized-processes').mkdir()
            actions = []
            class Process:
                def __init__(self, pid=1):
                    self.pid = pid
                def create_time(self):
                    return 2
                def cpu_affinity(self, values):
                    pass
            def spawn(command, **kwargs):
                name = Path(command[4]).name
                phase = command[6] if len(command) == 7 else None
                actions.append((name, phase))
                if name == 'verify_report.py':
                    put(base/(phase+'-verification.json'), dict(passed=True,
                        statistical_screen=aa_pass if phase == 'aa' else True, diagnostic_screen=True))
                code = collect_code if name == 'collect.py' else 0
                return types.SimpleNamespace(pid=3, wait=lambda **kwargs: code, poll=lambda: code)
            def observe(*args, **kwargs):
                return types.SimpleNamespace(returncode=0, stdout=json.dumps(dict(complete=True, code=0,
                    supervisor_live=False, latest=None)), stderr='')
            with contextlib.ExitStack() as stack:
                for key, value in dict(BASE=base, ROOT=root, MONITOR=monitor, WHISPER=whisper,
                                       CONTROL=control, SITE=root).items():
                    stack.enter_context(patch.object(finish, key, value))
                stack.enter_context(patch.dict(finish.sys.modules, psutil=types.SimpleNamespace(Process=Process)))
                stack.enter_context(patch.object(finish.subprocess, 'Popen', spawn))
                stack.enter_context(patch.object(finish.subprocess, 'run', observe))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if whisper_code or collect_code:
                    with self.assertRaises(AssertionError):
                        finish.main()
                else:
                    finish.main()
            state = json.loads((control/'state.json').read_text())
            self.assertTrue(state['complete'])
            self.assertEqual(state['code'], 1 if whisper_code or collect_code else 0)
            return actions

    def test_failed_aa_publishes_without_comparison(self):
        actions = self.scenario(False)
        self.assertEqual(actions, [('stage.py', None), ('collect.py', 'aa'), ('report.py', 'aa'),
                                  ('verify_report.py', 'aa'), ('update_benchmark.py', 'aa')])

    def test_passing_aa_starts_fixed_comparison_once(self):
        actions = self.scenario(True)
        self.assertEqual(actions.count(('start_compare.py', None)), 1)
        self.assertEqual(actions[-4:], [('collect.py', 'compare'), ('report.py', 'compare'),
                                      ('verify_report.py', 'compare'), ('update_benchmark.py', 'compare')])

    def test_failed_predecessor_or_collection_never_starts_comparison(self):
        self.assertEqual(self.scenario(True, whisper_code=1), [])
        self.assertEqual(self.scenario(True, collect_code=1), [('stage.py', None), ('collect.py', 'aa')])


if __name__ == '__main__':
    unittest.main()
