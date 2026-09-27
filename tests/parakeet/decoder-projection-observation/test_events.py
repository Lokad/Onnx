"""Exercise refusal/conservation logic; generated documents are never run evidence."""
import copy
from collections import Counter
import json
from pathlib import Path
import unittest
from events import CLR, CUSTOM, SAMPLE, calls, reconcile, stacks, method_records


def fixture():
    value = dict(passed=True, diagnostic_only=True, protocol='parakeet-decoder-projection-observation-v1',
        mode='trace', flags={}, controls=3, calls=1280, warmups=256, observations=1024, frequency=1000000,
        every_output_exact=True, inputs_unchanged=True, held_outputs_unchanged=True,
        initializers_unchanged=True, ordinary_outputs_only=True, pid=40, native_thread=41,
        counter_scope='Complete decoder execution; never per-node allocation or copy attribution', clocks=[])
    events = []

    def emit(provider, name, identifier, ms, payload):
        events.append(dict(index=len(events), provider=provider, name=name, id=identifier, ms=ms,
            payload=payload, pointerSize=8, rawBase64='', rawLength=0, pid=40, thread=41))

    for i in range(1280):
        marker = 1000 + 100 * i
        value['clocks'].append(dict(iteration=i, warmup=i < 256, marker=marker, start=marker+1,
            stop=marker+90, ticks=89, decoder_allocated_bytes=100, decoder_copy_bytes=0,
            decoder_scratch_bytes=0, decoder_gc_collections=[0, 0, 0]))
        emit(CUSTOM, 'Begin', 1, marker / 1000, dict(fixture='0', iteration=str(i), counter=str(marker)))
        emit(CUSTOM, 'End', 2, (marker + 90) / 1000, dict(fixture='0', iteration=str(i), counter=str(marker+90)))
    emit(CLR, 'Method/LoadVerbose', 143, 130., {})
    emit(SAMPLE, 'ThreadSample', 0, 130., {})
    summary = dict(complete=True, protocol='all-event-records-v1', lost=0, runtime='10.0.8',
        recorded=len(events), clr_events=1, allCounts=dict(Counter(e['provider']+':'+e['name'] for e in events)))
    return value, events, summary


def stack_fixture():
    def event(kind, at, frame): return dict(type=kind, at=at, frame=frame)
    document = {'$schema': 'https://www.speedscope.app/file-format-schema.json',
        'shared': {'frames': [{'name': 'Driver.Main'}, {'name': 'Lokad.Onnx.MathOps.mm_m1_kblocked'}, {'name': 'GC'}]},
        'profiles': [dict(name='Thread (41)', type='evented', unit='milliseconds', startValue=0., endValue=10.,
            events=[event('O', 0., 0), event('O', 1., 1), event('C', 9., 1), event('C', 10., 0)]),
            dict(name='Thread (42)', type='evented', unit='milliseconds', startValue=0., endValue=10.,
            events=[event('O', 0., 2), event('C', 10., 2)])]}
    intervals = [dict(begin_ms=2., end_ms=4.), dict(begin_ms=6., end_ms=8.)]
    return document, intervals


class EventTests(unittest.TestCase):
    def test_complete_marker_clock_reconciliation(self):
        value, events, summary = fixture()
        result = reconcile(value, events, summary)
        self.assertEqual(result['markers'], 2560)
        self.assertEqual(len(result['intervals']), 1280)
        self.assertEqual(result['decoder_allocated_bytes'], 128000)
        self.assertFalse(result['per_node_counters'])

    def test_lost_mixed_process_and_changed_marker_refused(self):
        for mutation in ['lost', 'pid', 'marker']:
            value, events, summary = fixture()
            if mutation == 'lost': summary['lost'] = 1
            elif mutation == 'pid': events[10]['pid'] = 99
            else: events[10]['payload']['counter'] = '1'
            with self.assertRaises(AssertionError): reconcile(value, events, summary)

    def test_incomplete_call_or_output_refused(self):
        for mutation in ['count', 'output', 'negative_counter']:
            value, _, _ = fixture()
            if mutation == 'count': value['clocks'].pop()
            elif mutation == 'output': value['every_output_exact'] = False
            else: value['clocks'][5]['decoder_copy_bytes'] = -1
            with self.assertRaises(AssertionError): calls(value)

    def test_stack_inside_outside_and_other_thread_conserved(self):
        document, intervals = stack_fixture()
        result = stacks(document, intervals, 41)
        self.assertEqual(sum(r['estimated_thread_ms'] for r in result['inside']), 4.)
        self.assertEqual(sum(r['estimated_thread_ms'] for r in result['outside']), 16.)
        self.assertEqual({r['category'] for r in result['inside']}, {'one-row-row-major'})
        self.assertFalse(result['per_call_jit_version_proved'])

    def test_invalid_stack_order_and_mixed_units_refused(self):
        for mutation in ['order', 'unit', 'missing_thread']:
            document, intervals = stack_fixture()
            if mutation == 'order': document['profiles'][0]['events'][2]['frame'] = 0
            elif mutation == 'unit': document['profiles'][0]['unit'] = 'seconds'
            else: document['profiles'][0]['name'] = 'Thread (1)'
            with self.assertRaises(AssertionError): stacks(document, intervals, 41)

    def test_untyped_rundown_retained(self):
        events = [dict(provider=CLR+'Rundown', name='EventID(144)', rawBase64=''),
                  dict(provider=CLR, name='Method/LoadVerbose'), dict(provider=CLR, name='GC/Start')]
        self.assertEqual(method_records(events), events[:2])

    def test_retained_original_stack_totals_reproduced(self):
        # Exercise the parser on an actual closed trace. This is an old, different
        # workload, not execution evidence for the pending decoder observation.
        root = Path(__file__).resolve().parents[3]
        base = root/'artifacts/parakeet-dispatch-events-amd-20260923'
        analysis = json.loads((base/'analysis.json').read_text())
        old = json.loads((root/'tests/parakeet/dispatch-events-results/analysis-20260923.json').read_text())
        for role in ['current', 'candidate']:
            value = json.loads((base/f'collected/{role}-capture/result.json').read_text())
            path, = (base/f'collected/{role}-stacks').glob('*.speedscope.json')
            result = stacks(json.loads(path.read_text()), analysis['reports'][role]['intervals'], value['nativeThread'])
            self.assertAlmostEqual(sum(r['estimated_thread_ms'] for r in result['inside']), old['reports'][role]['stack_inside_ms'], places=5)
            self.assertAlmostEqual(sum(r['estimated_thread_ms'] for r in result['outside']), old['reports'][role]['stack_outside_ms'], places=5)


if __name__ == '__main__': unittest.main()
