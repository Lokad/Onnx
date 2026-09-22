"""Reuse closed exact-runtime model/public gates, never old timing selection."""
from candidate_protocol import pin, read, gate

IDENTITIES = dict(
    production=('d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4',
        'e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'),
    portable=('e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838',
        '85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a'))
CLOSURES = dict(production='e0f1123a85c2a575e062fbf0e4dacc8c742c087661c2599d0197e5ca35cecc68',
    portable='d955ef2187a45e44e5d60bf92e17574384503570176f6fb592dcc1d9666a0f7a')


def qualification(base):
    reports = {}
    for role, identities in IDENTITIES.items():
        prior = base / 'retained' / role
        closed = read(prior / 'closed.json')
        assert pin(prior / 'closed.json')['sha256'] == CLOSURES[role] and closed['passed']
        assert pin(prior / 'analysis.json') == closed['files']['analysis.json']
        analysis = read(prior / 'analysis.json')
        assert analysis['passed']
        for family in ('pyannote', 'parakeet'):
            old = prior / (role + '-' + family + '.json')
            original_payload = read(prior / 'payload.json')
            assert pin(old) == original_payload['files']['manifests/' + old.name]
            assert pin(prior / 'payload.json') == closed['files']['collected/payload.json']
            current = base / 'manifests' / old.name
            # Preserve exact original manifests and their native/input identities.
            assert pin(current) == pin(old)
            manifest = read(current)
            assert (manifest['core_sha256'], manifest['data_sha256']) == identities
        for name, sha in zip(('Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'), identities, strict=True):
            assert pin(base / 'runtimes' / role / name)['sha256'] == sha
        for name in ('AudioBenchmark.dll', 'AudioBenchmark.deps.json', 'AudioBenchmark.runtimeconfig.json'):
            assert pin(base / 'runtimes' / role / name) == original_payload['files']['runtimes/' + role + '/' + name]
        reports[role] = analysis['reports'][role]
    gate(reports)
    return reports
