"""Recompute actual AMD caller and complete-model checks before timing."""
from candidate_protocol import ROLES, gate, pin, read
from qualify_outputs import pyannote, parakeet

IDENTITIES = dict(
    production=('1279b4b662241db2404fa4875eae20eaa924f15677a85655f99b8f81cd24b309',
                '4e602d9f6a35a51277d6deb9d75779d84cecf0a3a433d1b4eb70b0006462cca4'),
    portable=('3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f',
              '6318cf48691470b908eec4c4d09c558172e43ce3b04bca9039c68966998a684b'))


def retained_callers(base):
    folder = base/'retained-product'
    assert pin(folder/'closed.json')['sha256'] == '88ec9b71d6e7807440ccd65d6148ead8c585a0633ce755898e174b8501b77197'
    proof = read(folder/'closed.json'); assert proof['passed'] and proof['remote_terminal']
    assert pin(folder/'analysis.json') == proof['analysis'] == proof['files']['analysis.json']
    value = read(folder/'analysis.json')
    assert value['passed'] and value['core']['sha256'] == IDENTITIES['portable'][0]
    assert pin(base/'runtimes/portable/Lokad.Onnx.dll') == value['core']
    for width in ['256', '512']:
        raw = value['reports']['raw-'+width]
        assert raw['passed'] and raw['raw_cases'] == 2648 and raw['supplemental'] == 20 and raw['rejected'] == 10
        assert raw['control_requests'] == 2668 and raw['candidate_requests'] == 5336 and raw['fallback_graphs'] == 164
        assert raw['differences'] == raw['nan_payload_differences'] == raw['raw_nan_payload_differences'] == 0
        layer = value['reports']['layers-'+width]
        assert layer['passed'] and layer['cases'] == 108 and layer['values'] == 119823360 and layer['graph_calls'] == 216
        assert layer['native_maximum'] <= 1e-4 and layer['lanes'] == int(width)//32
    return dict(passed=True, closed=pin(folder/'closed.json'), analysis=pin(folder/'analysis.json'), reports=value['reports'])


def qualification(base, campaign):
    callers = retained_callers(base)
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
        reports[role] = dict(pyannote=graph, parakeet=parakeet(base, campaign / (role + '-parakeet.json'), role), callers=callers)
    gate(reports)
    return reports
