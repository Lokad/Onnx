"""Reject incomplete or contradictory trace evidence before VM execution."""
import base64
import struct
import unittest
from checks import CUSTOM, CLR, reconcile, codegen, codegen_bodies
from protocol import DISASM


def fixture():
    rows = []; entries = []
    events = [dict(index=0, provider=CLR, name='test', id=1, pid=7, thread=8, ms=0, rawBase64='', rawLength=0)]
    for case in range(21):
        entries.append(dict(name='fixture-' + str(case), node=case, m=50, k=1024, n=1024, y=dict(sha256='a' * 64)))
        row = dict(index=case, name=entries[-1]['name'], node=case, m=50, reduction=1024, columns=1024,
            exact=True, inputs=True, guards=True, preparationTicks=1, output='a' * 64,
            addressesBefore=dict(a=1, b=2, c=3), addressesAfter=dict(a=1, b=2, c=3), clocks=[])
        rows.append(row)
        for i in range(120):
            counter = 100 * (case * 120 + i) + 1
            row['clocks'].append(dict(iteration=i, warmup=i < 60, marker=counter, start=counter + 1, stop=counter + 11, ticks=10,
                allocated=i, allocatedAfter=i + 1, gc0=0, gc1=0, gc2=0, after0=0, after1=0, after2=0))
            for identifier, count in [(1, counter), (2, counter + 11)]:
                raw = struct.pack('<iiq', case, i, count)
                events.append(dict(index=len(events), provider=CUSTOM, name=str(identifier), id=identifier,
                    pid=7, thread=8, ms=count, rawBase64=base64.b64encode(raw).decode(), rawLength=16,
                    payload=dict(fixture=str(case), iteration=str(i), counter=str(count))))
    return dict(passed=True, diagnosticOnly=True, protocol='parakeet-dispatch-events-v1', calls=2520,
        warmups=1260, measured=1260, frequency=1000, rows=rows, pid=7, nativeThread=8), events, dict(
        complete=True, lost=0, clr_events=1, protocol='all-event-records-v1', recorded=len(events),
        allCounts={CLR + ':test': 1, CUSTOM + ':1': 2520, CUSTOM + ':2': 2520}), dict(entries=entries)


class TraceChecks(unittest.TestCase):
    def test_complete_census(self):
        value, events, summary, capture = fixture()
        result = reconcile(value, events, summary, capture)
        self.assertEqual((len(result['calls']), len(result['blocks']), result['markers']), (2520, 126, 5040))
        self.assertEqual((result['blocks'][-1]['case'], result['blocks'][-1]['last']), (20, 119))
        self.assertEqual(result['blocks'][0]['total_allocated_bytes'], 1)

    def test_loss_and_missing_event(self):
        value, events, summary, capture = fixture(); summary['lost'] = 1
        with self.assertRaises(AssertionError): reconcile(value, events, summary, capture)
        summary['lost'] = 0
        with self.assertRaises(AssertionError): reconcile(value, events[:-1], summary, capture)
        summary['allCounts'][CLR + ':test'] = 2
        with self.assertRaises(AssertionError): reconcile(value, events, summary, capture)

    def test_marker_thread_process_and_decoding(self):
        for key, bad in [('thread', 9), ('pid', 9), ('rawLength', 15), ('ms', float('nan'))]:
            with self.subTest(key=key):
                value, events, summary, capture = fixture(); events[12][key] = bad
                with self.assertRaises(AssertionError): reconcile(value, events, summary, capture)
        for raw in [(1, 5, 512), (0, 5, 999), (0, 6, 512)]:
            value, events, summary, capture = fixture()
            events[12]['rawBase64'] = base64.b64encode(struct.pack('<iiq', *raw)).decode()
            with self.assertRaises(AssertionError): reconcile(value, events, summary, capture)
        value, events, summary, capture = fixture(); events[12]['payload']['counter'] = '999'
        with self.assertRaises(AssertionError): reconcile(value, events, summary, capture)

    def test_clocks_collection_allocation(self):
        for key, bad in [('ticks', 11), ('after2', -1), ('allocatedAfter', 0), ('warmup', False), ('iteration', 3)]:
            with self.subTest(key=key):
                value, events, summary, capture = fixture(); value['rows'][1]['clocks'][20][key] = bad
                with self.assertRaises(AssertionError): reconcile(value, events, summary, capture)

    def test_fixtures_guards_and_addresses(self):
        for key, bad in [('m', 51), ('name', 'wrong'), ('output', 'b' * 64), ('guards', False), ('inputs', False), ('exact', False), ('addressesAfter', dict(a=0,b=2,c=3))]:
            with self.subTest(key=key):
                value, events, summary, capture = fixture(); value['rows'][2][key] = bad
                with self.assertRaises(AssertionError): reconcile(value, events, summary, capture)


def listing(candidate=False):
    names = ['MatMul2DCore', '<MatMul2DCore>b__0', 'RunBatchedFloatMatMul', '<RunBatchedFloatMatMul>b__1', 'RunFloatMatMulKernel']
    if candidate: names += ['RunIsolatedShortWideKernel', 'RunIsolatedShortWidePackedRows', 'ShortWidePackPanelsB', 'ShortWideMultiply2Rows', 'ShortWideMultiply3Rows', 'ShortWideMultiplyRemainder']
    return ''.join('; Assembly listing for method Lokad.Onnx.Tensor`1:' + name + '():void (Tier0)\n'
        'G_M000_IG01:\n       jmp G_M000_IG02\nG_M000_IG02:\n       ret\n; Total bytes of code 3\n' for name in names)


class CodegenChecks(unittest.TestCase):
    def test_complete_bodies_and_probes(self):
        self.assertEqual(len(codegen_bodies(listing(), 'current')), 5)
        self.assertEqual(len(codegen_bodies(listing(True), 'candidate')), 11)
        value, _, _, capture = fixture()
        value.update(codegenOnly=True, protocol='parakeet-isolated-codegen-v1', flags={'DOTNET_JitDisasm': DISASM},
            callerProbes=dict(parallel2d=80, sequentialBatch=80, parallelBatch=80, exact=True))
        codegen(value, capture)
        value['callerProbes']['parallelBatch'] = 79
        with self.assertRaises(AssertionError): codegen(value, capture)
        value['callerProbes']['parallelBatch'] = 80; value['flags']['DOTNET_TieredCompilation'] = '0'
        with self.assertRaises(AssertionError): codegen(value, capture)

    def test_incomplete_bodies_and_missing_routes(self):
        for text, role in [(listing()[:-24], 'current'), (listing(), 'candidate'),
                           (listing().replace('<MatMul2DCore>b__0', 'wrong'), 'current'),
                           (listing().replace('jmp G_M000_IG02', 'jmp G_M000_IG99'), 'current')]:
            with self.subTest(role=role):
                with self.assertRaises(AssertionError): codegen_bodies(text, role)


if __name__ == '__main__': unittest.main()
