"""Recompute actual AMD caller and complete-model checks before timing."""
from candidate_protocol import ROLES, gate, pin, read
from qualify_outputs import pyannote, parakeet

IDENTITIES = dict(
    production=('1279b4b662241db2404fa4875eae20eaa924f15677a85655f99b8f81cd24b309',
                '4e602d9f6a35a51277d6deb9d75779d84cecf0a3a433d1b4eb70b0006462cca4'),
    portable=('abbf5e9878aeccf929a327ef4c1c3f5c9ff8afb8d9dec7ddb80455636cd1684d',
              'eb452663a09daa5d287fff1f65c07f2105f5c10ab474f845f210b2b30e3b921f'))


def caller(base, campaign, mode):
    value = read(campaign / ('caller-' + mode + '.json'))
    assert value['passed'] and value['mode'] == mode and value['runtime'] == '10.0.8'
    assert value['candidate'] == IDENTITIES['portable'][0] and value['baseline'] == IDENTITIES['production'][0]
    assert value['processor_count'] == 1 and value['fma'] == (mode == 'normal')
    assert value['flags'] == ([] if mode == 'normal' else ['DOTNET_EnableHWIntrinsic'])
    state = read(campaign / 'identity.json')
    owner = next(r for r in state['runs'] if r['name'] == 'caller-' + mode)
    assert owner['code'] == 0 and value['pid'] == owner['child']['pid']
    expected = [(s['m'], s['n'], s['k'], 1, True, 'finite', 'Auto', 0)
                for s in read(base / 'caller/shapes.json')['shapes']]
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
    rows = value['records']
    assert len(rows) == len(expected) == 400
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
    return dict(cases=400, payload_differences=payloads, output=pin(campaign / ('caller-' + mode + '.json')))


def qualification(base, campaign):
    callers = {mode: caller(base, campaign, mode) for mode in ['normal', 'disabled']}
    from operator_checks import probe
    probes = {mode: probe(read(campaign / ('probe-' + mode + '.json')), mode, '10.0.8', mode == 'normal')
              for mode in ['normal', 'disabled']}
    reports = {}
    for role in ROLES:
        for name, sha in zip(['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'], IDENTITIES[role], strict=True):
            assert pin(base / 'runtimes' / role / name)['sha256'] == sha
        baseline = None if role == 'production' else campaign / 'production-pyannote-output'
        graph = pyannote(base, campaign / (role + '-pyannote-output'), role, baseline)
        if role == 'portable':
            assert all(r['bit_identical'] for r in graph['comparisons'] if r['reference'] == 'production')
            current = read(campaign / 'portable-pyannote-output/result.json')['applications']
            previous = read(campaign / 'production-pyannote-output/result.json')['applications']
            assert len(current) == len(previous) == 16
            assert all(a['result'] == b['result'] for a, b in zip(current, previous, strict=True))
        reports[role] = dict(pyannote=graph, parakeet=parakeet(base, campaign / (role + '-parakeet.json'), role), callers=callers, arithmetic_probes=probes)
    gate(reports)
    return reports
