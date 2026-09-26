"""Audit every priming marker, then reuse the unchanged suffix/resource auditor."""
import base64
import json
from pathlib import Path
import struct
import sys
from prepare import ROOT, BASE, OLD, previous_closed
from protocol import pin, read


def prefix_checked(prefix, value, events):
    assert prefix['passed'] and prefix['protocol'] == 'pad-census-ten-seconds-after-first-v1'
    assert prefix['pid'] == value['pid'] and prefix['nativeThread'] == value['nativeThread']
    assert prefix['frequency'] == value['frequency'] > 0
    frequency = prefix['frequency']
    rounds = prefix['passes']
    assert 2 <= len(rounds) <= 16 and prefix['calls'] == len(rounds) * 9360
    assert prefix['began'] < rounds[0]['start']
    assert prefix['firstEnd'] == rounds[0]['end'] and prefix['ended'] == rounds[-1]['end']
    assert 0 < prefix['ended'] - prefix['began'] < 180 * frequency
    assert prefix['ended'] - prefix['firstEnd'] >= 10 * frequency
    assert rounds[-2]['end'] - prefix['firstEnd'] < 10 * frequency
    markers = [e for e in events if e['provider'] == 'Lokad-Parakeet-Pad-Diagnostic' and e['id'] in (3, 4)]
    assert len(markers) == prefix['calls'] * 2
    assert all(a['ms'] <= b['ms'] for a, b in zip(markers, markers[1:]))
    assert all(e['pid'] == value['pid'] and e['thread'] == value['nativeThread'] for e in markers)
    offset = 0; prior = prefix['began']
    for round_index, group in enumerate(rounds):
        assert group['round'] == round_index and prior < group['start']
        assert len(group['rows']) == 12
        prior = group['start']
        for case, (row, expected) in enumerate(zip(group['rows'], value['rows'], strict=True)):
            for key in ('index', 'name', 'shape', 'pads', 'mode', 'fill', 'output', 'exact', 'inputs', 'ownership'):
                assert row[key] == expected[key], key
            assert row['setupTicks'] > 0 and len(row['clocks']) == 780
            for iteration, clock in enumerate(row['clocks']):
                assert clock['iteration'] == iteration and clock['warmup'] == (iteration < 600)
                assert prior < clock['marker'] <= clock['start'] < clock['stop']
                assert clock['ticks'] == clock['stop'] - clock['start']
                assert clock['allocatedAfter'] >= clock['allocated'] >= 0
                assert clock['totalAllocatedAfter'] >= clock['totalAllocated'] >= 0
                assert all(clock['after' + str(g)] >= clock['gc' + str(g)] >= 0 for g in range(3))
                for marker, identifier, counter in zip(markers[offset:offset + 2], (3, 4),
                        (clock['marker'], clock['stop']), strict=True):
                    number = round_index * 780 + iteration
                    assert marker['id'] == identifier
                    assert struct.unpack('<iiq', base64.b64decode(marker['rawBase64'], validate=True)) == (case, number, counter)
                    assert {k: int(v) for k, v in marker['payload'].items()} == dict(fixture=case, iteration=number, counter=counter)
                offset += 2; prior = clock['stop']
        assert prior < group['end']; prior = group['end']
    assert prefix['ended'] < value['rows'][0]['clocks'][0]['marker']
    suffix = [e for e in events if e['provider'] == 'Lokad-Parakeet-Pad-Diagnostic' and e['id'] in (1, 2)]
    assert markers[-1]['ms'] < suffix[0]['ms']
    return dict(passed=True, rounds=len(rounds), calls=prefix['calls'], markers=len(markers),
        seconds=(prefix['ended'] - prefix['began']) / frequency,
        seconds_after_first=(prefix['ended'] - prefix['firstEnd']) / frequency,
        first_ms=markers[0]['ms'], last_ms=markers[-1]['ms'])


def main():
    sys.path.append(str(OLD))
    scope = dict(__name__='retained_suffix_auditor', __file__=str(OLD / 'audit.py'))
    exec(compile((OLD / 'audit.py').read_text(encoding='utf8'), str(OLD / 'audit.py'), 'exec'), scope)
    original_reconcile = scope['reconcile']
    original_save = scope['save']
    prefixes = {}; states = {}

    def reconcile(value, events, summary):
        role = value['role']
        prefix = read(BASE / 'collected' / (role + '-capture') / 'priming.json')
        prefixes[role] = prefix_checked(prefix, value, events)
        result = original_reconcile(value, events, summary)
        methods = ('Pad', 'PadCore') + (('PadDispatch',) if role == 'candidate' else ())
        loads = [dict(ms=e['ms'], method=e['payload']['MethodName'], tier=e['payload']['OptimizationTier'],
                      bytes=e['payload']['MethodSize'], address=e['payload']['MethodStartAddress'])
                 for e in events if e['provider'] == 'Microsoft-Windows-DotNETRuntime'
                 and e['name'] == 'Method/LoadVerbose'
                 and e['payload'].get('MethodNamespace') == 'Lokad.Onnx.CPUExecutionProvider'
                 and e['payload'].get('MethodName') in methods]
        first = result['calls'][600]['begin_ms']
        available = {method: any(e['method'] == method and e['tier'] == 'OptimizedTier1'
                                and e['ms'] < first for e in loads) for method in methods}
        states[role] = dict(full_methods_before_first_measurement=available,
            available=all(available.values()), first_measurement_ms=first, loads=loads,
            later_loads=[e for e in loads if e['ms'] >= first])
        return result

    def save(path, value):
        if path.name == 'analysis.json':
            assert set(prefixes) == set(states) == {'current', 'candidate'}
            value.update(priming=prefixes, compilation_state=states,
                state_condition_met=all(row['available'] for row in states.values()),
                admitted=False, diagnostic_protocol='pad-census-ten-seconds-after-first-v1')
        elif path.name == 'closed.json':
            value.update(admitted=False, original_screens_remain_rejected=True)
        original_save(path, value)

    scope.update(BASE=BASE, previous_closed=previous_closed, reconcile=reconcile, save=save)
    scope['main']()


if __name__ == '__main__': main()
