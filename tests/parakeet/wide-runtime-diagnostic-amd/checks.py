"""Reconcile full matrix diagnostics without scoring their clocks."""
import base64
import collections
import math
import re
import struct
from protocol import DISASM

CUSTOM = 'Lokad-Parakeet-MatMul-Diagnostic'
CLR = 'Microsoft-Windows-DotNETRuntime'


def rows(value, capture):
    assert value['passed'] and value['diagnosticOnly']
    assert value['calls'] == 2520 and value['warmups'] == value['measured'] == 1260
    assert len(value['rows']) == len(capture['entries']) == 21
    assert type(value['frequency']) is int and value['frequency'] > 0
    for i, (row, fixture) in enumerate(zip(value['rows'], capture['entries'], strict=True)):
        assert row['index'] == i
        assert [row[k] for k in ['name', 'node', 'm', 'reduction', 'columns']] == [fixture[k] for k in ['name', 'node', 'm', 'k', 'n']]
        assert row['exact'] and row['guards'] and row['inputs'] and row['output'] == fixture['y']['sha256']
        assert row['preparationTicks'] > 0 and len(row['clocks']) == 120
        for j, clock in enumerate(row['clocks']):
            assert clock['iteration'] == j and clock['warmup'] == (j < 60)
            assert type(clock['ticks']) is int and clock['ticks'] > 0


def reconcile(value, events, summary, capture):
    rows(value, capture)
    assert value['protocol'] == 'parakeet-dispatch-events-v1'
    assert summary['complete'] and summary['lost'] == 0 and summary['clr_events'] > 0
    assert summary['protocol'] == 'all-event-records-v1' and summary['recorded'] == len(events)
    assert [e['index'] for e in events] == list(range(len(events)))
    assert dict(collections.Counter(e['provider'] + ':' + e['name'] for e in events)) == summary['allCounts']
    assert sum(e['provider'] == CLR for e in events) == summary['clr_events']
    assert all(math.isfinite(e['ms']) and e['ms'] >= 0 for e in events)
    for event in events: assert len(base64.b64decode(event['rawBase64'], validate=True)) == event['rawLength']
    markers = [e for e in events if e['provider'] == CUSTOM and e['id'] in [1, 2]]
    assert len(markers) == 5040
    assert all(e['pid'] == value['pid'] and e['thread'] == value['nativeThread'] for e in markers)
    assert all(a['ms'] <= b['ms'] for a, b in zip(markers, markers[1:]))
    prior = 0; calls = []; blocks = []
    for case_index, row in enumerate(value['rows']):
        for key in ['addressesBefore', 'addressesAfter']:
            assert set(row[key]) == {'a', 'b', 'c'} and all(type(v) is int and v > 0 for v in row[key].values())
        for index, clock in enumerate(row['clocks']):
            assert prior < clock['marker'] <= clock['start'] < clock['stop']
            assert clock['ticks'] == clock['stop'] - clock['start']; prior = clock['stop']
            assert clock['allocatedAfter'] >= clock['allocated'] >= 0
            assert all(clock['after' + str(g)] >= clock['gc' + str(g)] >= 0 for g in range(3))
            pair = markers[len(calls) * 2:len(calls) * 2 + 2]
            for event, identifier, counter in zip(pair, [1, 2], [clock['marker'], clock['stop']], strict=True):
                assert event['id'] == identifier
                assert struct.unpack('<iiq', base64.b64decode(event['rawBase64'])) == (case_index, index, counter)
                assert {k: int(v) for k, v in event['payload'].items()} == dict(fixture=case_index, iteration=index, counter=counter)
            calls.append(dict(index=len(calls), case=case_index, name=row['name'], iteration=index, warmup=index < 60,
                begin_ms=pair[0]['ms'], end_ms=pair[1]['ms'], wall_ms=clock['ticks'] * 1000 / value['frequency'],
                total_allocated_bytes=clock['allocatedAfter'] - clock['allocated'],
                gc0=clock['after0'] - clock['gc0'], gc1=clock['after1'] - clock['gc1'], gc2=clock['after2'] - clock['gc2']))
        case_calls = calls[-120:]
        for start in range(0, 120, 20):
            group = case_calls[start:start + 20]
            blocks.append(dict(case=case_index, name=row['name'], first=start, last=start + 19, warmup=start < 60,
                wall_ms=sum(c['wall_ms'] for c in group) / 20,
                total_allocated_bytes=sum(c['total_allocated_bytes'] for c in group) / 20,
                gc_calls=sum(any(c[k] for k in ['gc0', 'gc1', 'gc2']) for c in group)))
    return dict(calls=calls, blocks=blocks, markers=len(markers), events=len(events), clr_events=summary['clr_events'])


def codegen(value, capture):
    rows(value, capture)
    assert value['codegenOnly'] and value['protocol'] == 'parakeet-isolated-codegen-v1'
    assert value['flags'] == {'DOTNET_JitDisasm': DISASM}
    assert value['callerProbes'] == dict(parallel2d=80, sequentialBatch=80, parallelBatch=80, exact=True)


def codegen_bodies(text, role):
    matches = list(re.finditer(r'^; Assembly listing for method ([^\n]+) \(([^\n]+)\)\n(.*?); Total bytes of code (\d+)', text, re.M | re.S))
    assert matches and len(matches) == text.count('; Assembly listing for method ') == text.count('; Total bytes of code ')
    methods = [m[1] for m in matches]
    for name in ['MatMul2DCore(', '<MatMul2DCore>b__0(', 'RunBatchedFloatMatMul(', '<RunBatchedFloatMatMul>b__1(', 'RunFloatMatMulKernel(']:
        assert any(name in m for m in methods), name
    if role == 'candidate':
        for name in ['RunIsolatedShortWideKernel(', 'RunIsolatedShortWidePackedRows(', 'ShortWidePackPanelsB(', 'ShortWideMultiply2Rows(', 'ShortWideMultiply3Rows(', 'ShortWideMultiplyRemainder(']:
            assert any(name in m for m in methods), name
    result = []
    for index, match in enumerate(matches):
        labels = re.findall(r'^(G_M\d+_IG\d+):', match[0], re.M)
        assert labels and len(labels) == len(set(labels)) and set(re.findall(r'G_M\d+_IG\d+', match[0])) <= set(labels)
        result.append(dict(index=index, method=match[1], tier=match[2], bytes=int(match[4]), labels=len(labels),
            calls=[s.strip() for s in match[3].splitlines() if re.match(r'^\s+(call|tail\.jmp)\s', s)]))
    return result
