"""Check that nested timing and attribution cannot silently lose or duplicate work."""
import ast
from pathlib import Path
import unittest
from analyze import exclusive, profile_graph
from observer import instrument


def node(name, start, duration):
    return dict(name=name+'_kernel_time', cat='Node', ph='X', pid=1, tid=2,
                ts=start, dur=duration, args=dict(provider='CPUExecutionProvider', op_name=name))


class Attribution(unittest.TestCase):
    def test_nested_time_counted_once(self):
        values = exclusive([node('parent', 0, 100), node('child', 10, 60), node('leaf', 20, 20)])
        self.assertEqual([duration for _, duration in values], [40, 40, 20])
        self.assertEqual(sum(duration for _, duration in values), 100)

    def test_partial_overlap_refused(self):
        with self.assertRaises(AssertionError):
            exclusive([node('a', 0, 20), node('b', 10, 20)])

    def test_warmup_separated_and_unassigned_refused(self):
        events = [dict(name='model_run', ph='X', ts=t, dur=100, pid=1, tid=2) for t in [0, 200]]
        events += [node('MatMul', 10, 80), node('MatMul', 210, 40)]
        calls = [dict(request=i) for i in range(2)]
        requests = [dict(iteration=i) for i in range(2)]
        result = profile_graph(events, calls, requests)
        self.assertEqual(result['node_clocks'][0]['exclusive_us'], 40)
        self.assertEqual(result['phase_totals']['warmup']['node_us'], 80)
        with self.assertRaises(AssertionError):
            profile_graph(events+[node('lost', 500, 10)], calls, requests)

    def test_missing_session_call_refused(self):
        with self.assertRaises(AssertionError):
            profile_graph([], [dict(request=0)], [dict(iteration=0)])

    def test_exact_single_consumer_insertion(self):
        root = Path(__file__).resolve().parents[3]
        source = (root/'artifacts/parakeet-winograd-baseline-amd-20260923/collected/runtime/native.py').read_text()
        compile(instrument(source), 'native.py', 'exec')
        with self.assertRaises(AssertionError):
            instrument(source.replace('module_spec.loader.exec_module(module)', 'pass'))


if __name__ == '__main__':
    unittest.main()
