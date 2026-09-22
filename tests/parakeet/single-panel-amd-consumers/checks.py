"""Coverage and identity checks shared by local and AMD consumer qualification."""
from adapt import CORE, CONTROL_CORE


def probe(value, mode, runtime, avx512):
    assert value['passed'] and value['mode'] == mode and value['runtime'] == runtime
    assert value['core_sha256'] == CORE and value['original_core_sha256'] == CONTROL_CORE
    assert value['affinity'] == 4 and value['processor_count'] == 1 and value['avx512'] == avx512
    keys = ['tests', 'eligible', 'fallback', 'changed', 'raw_contracts', 'dynamic_dispatch_cases', 'actual_operand_controls']
    assert tuple(value[k] for k in keys) == ((416, 64, 352, 64, 416, 48, 4) if mode == 'normal' else (4, 0, 0, 0, 0, 0, 0))
    expected = [(m, n, k) for m in [8, 9, 12, 13, 14, 16, 17, 20, 24, 25] for n in [1024, 1025] for k in [32, 64]] if avx512 else []
    rows = value['prepared_precedence']
    assert [(r['m'], r['n'], r['k']) for r in rows] == expected
    for row in rows:
        assert row['baseline'] == row['candidate']
        assert row['differs'] == (row['partial'] != row['candidate'])
        assert row['guards_preserved'] and row['operands_preserved']
        for name in ['baseline', 'candidate', 'partial', 'input_a', 'input_b', 'packed', 'initial']:
            assert len(row[name]) == 64 and all(c in '0123456789abcdef' for c in row[name])
    assert not avx512 or any(r['differs'] for r in rows)
    return dict(original_cases=value['tests'], prepared_precedence_cases=len(rows), distinct_partial_results=sum(r['differs'] for r in rows))


def caller(value, shapes, mode, runtime):
    assert value['passed'] and value['mode'] == mode and value['runtime'] == runtime
    assert value['candidate'] == CORE and value['baseline'] == CONTROL_CORE
    assert value['processor_count'] == 1 and value['fma'] == (mode == 'normal')
    assert value['flags'] == ([] if mode == 'normal' else ['DOTNET_EnableHWIntrinsic'])
    expected = [(s['m'], s['n'], s['k'], 1, True, 'finite', 'Auto', 0) for s in shapes['shapes']]
    for m in [32, 64]:
        for n in [64, 65]:
            for k in [1, 2, 7, 8, 31, 32, 33]:
                for bias in [False, True]:
                    for pattern in ['finite', 'zero', 'special']:
                        expected.extend((m, n, k, 2, bias, pattern, 'Auto', p) for p in [0, 1])
    for policy in ['Scalar', 'Simd']:
        for k in [2, 8, 33]:
            for bias in [False, True]:
                for pattern in ['finite', 'zero', 'special']:
                    expected.append((32, 64, k, 2, bias, pattern, policy, 0))
    expected.extend((m, n, k, 2, True, 'finite', 'Auto', 0)
        for m, n, k in [(5, 9, 2), (30, 64, 8), (33, 64, 2), (96, 64, 8), (32, 63, 32), (32, 1024, 65)])
    rows = value['records']; assert len(rows) == len(expected) == 400
    assert [tuple(r[k] for k in ['rows', 'reduction', 'block', 'groups', 'hasBias', 'pattern', 'policy', 'pass']) for r in rows] == expected
    payloads = 0
    for r in rows:
        assert r['checked_values'] == 2*r['rows']*r['groups']*(2*r['block']+1)+6
        assert len(r['digest']) == len(r['baseline_digest']) == 64
        assert 0 <= r['nan_values'] <= r['checked_values']
        for d in r['nan_payload_differences']:
            left, right = int(d['baseline'], 16), int(d['candidate'], 16)
            assert left != right and all((v & 0x7f800000) == 0x7f800000 and (v & 0x7fffff) != 0 for v in [left, right])
            assert d['batch'] in [0, 1] and 3 <= d['index'] < r['checked_values'] - 3
        payloads += len(r['nan_payload_differences'])
        if r['pattern'] != 'special':
            assert r['nan_values'] == 0 and r['digest'] == r['baseline_digest'] and not r['nan_payload_differences']
    return dict(cases=400, payload_differences=payloads)
