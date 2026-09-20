import unittest
import copy
from common import *
from routes import *

class References(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        psutil_module().Process().cpu_affinity([0])
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cls.window = np.load(PRIOR / 'reference/window.npy')
        cls.mel = np.load(PRIOR / 'reference/mel.npy')
        cls.tables = direct_tables()

    def test_boundaries_and_independent_stages(self):
        for count in (400, 559, 560, 561, 720):
            for kind in ('noise', 'quiet', 'silence', 'dc', 'impulse'):
                samples = np.random.default_rng(20260920).uniform(-.2, .2, count).astype(np.float32)
                if kind == 'quiet': samples *= np.float32(1e-6)
                if kind in ('silence', 'dc'): samples.fill(0 if kind == 'silence' else .25)
                if kind == 'impulse': samples.fill(0); samples[count // 2] = .5
                before = samples.copy()
                left = numpy_route(samples, self.window, self.mel)
                right = torch_route(samples, self.window, self.mel)
                frames = 1 + (count - 400) // 160
                self.assertEqual(left['features'].shape, (1, frames, 80))
                np.testing.assert_array_equal(samples, before)
                for stage in STAGES:
                    self.assertEqual(metric(left[stage], right[stage], REFERENCE_LIMIT)['failed'], 0, (count, kind, stage))
                for frame in range(frames):
                    scalar = scalar_window(samples, self.window, frame)
                    real, imaginary = direct_fourier(scalar, self.tables)
                    for route in (left, right):
                        for stage, expected in [('windowed', scalar), ('real', real), ('imaginary', imaginary)]:
                            self.assertEqual(metric(route[stage][frame], expected, REFERENCE_LIMIT)['failed'], 0, (count, kind, stage))
                if frames == 1 or kind in ('silence', 'dc'):
                    self.assertLessEqual(float(np.abs(left['features']).max()), 1e-12)
                self.assertLess(float(np.abs(left['features'].mean(axis=1)).max()), 1e-12)

    def test_invalid_contracts(self):
        valid = np.zeros(400, np.float32)
        bad = [np.zeros(399, np.float32), np.zeros(480001, np.float32), np.zeros(400, np.float64),
               np.zeros((1, 400), np.float32), np.full(400, np.nan, np.float32), np.full(400, 1.1, np.float32)]
        for route in (numpy_route, torch_route):
            for value in bad:
                with self.assertRaises(AssertionError): route(value, self.window, self.mel)
            with self.assertRaises(AssertionError): route(valid, self.window[:-1], self.mel)
            with self.assertRaises(AssertionError): route(valid, self.window, self.mel[:, :-1])
            with self.assertRaises(AssertionError): route(valid, self.window, -np.ones_like(self.mel))

    def test_resource_refusals(self):
        from audit import resource_checks
        supervisor = dict(pid=1, birth=100)
        run = dict(engine='numpy', complete=True, code=0, seconds=1, started=101, ended=102,
                   worker=dict(pid=2, birth=101), preflight_available=LIMITS['preflight'], preflight_disk=LIMITS['disk'], samples=2)
        samples = [dict(seconds=t, pid=2, birth=101, affinity=[0], rss=1000000, available=LIMITS['available']) for t in [.1, .6]]
        resource_checks(run, samples, supervisor)
        for name, value in [('complete', False), ('code', 1), ('seconds', 181), ('preflight_available', 0), ('preflight_disk', 0), ('samples', 3)]:
            changed = copy.deepcopy(run); changed[name] = value
            with self.assertRaises(AssertionError): resource_checks(changed, samples, supervisor)
        for name, value in [('pid', 3), ('birth', 100), ('affinity', [2]), ('rss', LIMITS['rss']), ('available', 0), ('seconds', 2)]:
            changed = copy.deepcopy(samples); changed[0][name] = value
            with self.assertRaises(AssertionError): resource_checks(run, changed, supervisor)

if __name__ == '__main__':
    unittest.main()
