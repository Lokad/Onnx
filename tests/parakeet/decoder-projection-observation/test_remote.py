"""Verify bounded commands and unmodified inherited supervisor/resource checks."""
import inspect
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import protocol

# These local tests inspect commands and pure resource checks. The Linux worker
# uses the already installed VM psutil. An empty import placeholder makes any
# attempted process observation fail; it does not simulate a live process.
if importlib.util.find_spec('psutil') is None:
    with patch.dict(sys.modules, {'psutil': ModuleType('psutil')}):
        import remote
else:
    import remote


class RemoteTests(unittest.TestCase):
    def test_only_original_graph_observer_runs(self):
        for name, mode in [('control-run', 'control'), ('trace-capture', 'trace')]:
            command, build, cpu = remote.command_for(name, dict(feed='/frozen/feed'))
            self.assertFalse(build)
            self.assertEqual(cpu, 2)
            self.assertEqual(command[-2], mode)
            self.assertEqual(Path(command[1]).name, 'ParakeetDecoderObservation.dll')
            self.assertEqual(Path(command[2]).name, 'observation.json')
            self.assertFalse(any(str(v).startswith(('DOTNET_', 'COMPlus_', 'LOKAD_')) for v in command))

    def test_offline_build_and_existing_exporters(self):
        for name in ['observer-restore', 'observer-build']:
            command, build, cpu = remote.command_for(name, dict(feed='/frozen/feed'))
            self.assertTrue(build)
            self.assertEqual(cpu, 2)
            self.assertIn('--tl:off', command)
            if name.endswith('restore'):
                self.assertEqual(command[command.index('--source') + 1], '/frozen/feed')
            else:
                self.assertIn('--no-restore', command)
        for name in ['trace-export', 'trace-stacks', 'tracer-version']:
            command, build, cpu = remote.command_for(name, {})
            self.assertFalse(build)
            self.assertEqual(cpu, 0)
        self.assertEqual(len(protocol.JOBS), 8)
        self.assertEqual(protocol.LIMITS['seconds'], 300)
        self.assertEqual(protocol.LIMITS['rss'], 4*1024**3)

    def test_supervisor_and_resource_checker_reused(self):
        self.assertIs(remote.main, remote.worker.main)
        self.assertIs(protocol.check_sample, protocol.inherited.check_sample)
        self.assertIs(remote.worker.command_for, remote.command_for)
        self.assertIs(remote.worker.after, remote.after)
        self.assertEqual(remote.worker.LIMITS, protocol.LIMITS)
        self.assertEqual(remote.worker.PROVIDERS, protocol.PROVIDERS)
        self.assertEqual(Path(inspect.getsourcefile(remote.main)), protocol.PARENT/'remote.py')
        self.assertEqual(Path(inspect.getsourcefile(protocol.check_sample)), protocol.PARENT/'protocol.py')

    def test_monitor_refuses_wrong_cpu_and_memory_overrun(self):
        good = dict(seconds=1., rss=1024, available=2*1024**3, tmpfs=4*1024**3,
            output=0, artifacts=0, members=[dict(rss=1024, expected_affinity=[2], affinity=[2], threads=[dict(affinity=[2])])])
        protocol.check_sample(good)
        good['members'][0]['threads'][0]['affinity'] = [0]
        with self.assertRaises(AssertionError): protocol.check_sample(good)
        good['members'][0]['threads'][0]['affinity'] = [2]
        good['available'] = 0
        with self.assertRaises(AssertionError): protocol.check_sample(good)


if __name__ == '__main__': unittest.main()
