"""Derive a fresh campaign from the closed portable integration protocol."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'portable-amd-integration'
NAMES = ['candidate_protocol.py', 'admission.py', 'transport.py', 'finish.py',
         'supervise.py', 'audit_results.py', 'qualify_outputs.py', 'meeting_protocol.py',
         'meetings_audit.py', 'prepare_execution.py', 'test_protocol.py', 'test_finish.py', 'test_admission.py']


def replace(text, old, new, count):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


for name in NAMES:
    target = HERE / name
    assert not target.exists(), target
    text = (OLD / name).read_text(encoding='utf8')
    if name == 'transport.py':
        for old, new in [('pyannote-portable-amd-payload-20260922', 'pyannote-direct-amd-payload-20260922'),
                         ('pyannote-portable-amd-execution-20260922', 'pyannote-direct-amd-execution-20260922'),
                         ('lokad-pyannote-portable-integration-20260922', 'lokad-pyannote-direct-20260922')]:
            text = replace(text, old, new, 1)
    if name == 'admission.py':
        text = replace(text, 'root production (Cored1f/Datae7)', 'selected production (Coree9c/Data85)', 1)
        text = replace(text, 'current portable-only (Coree9c/Data85)', 'direct-output candidate (Core19b/Data cb6)', 1)
    if name == 'supervise.py':
        text = replace(text, "('Lokad.Onnx.dll', 3108, True)", "('Lokad.Onnx.dll', 3113, True)", 1)
        text = replace(text, "3290 if name == 'backend' else 342", "3311 if name == 'backend' else 343", 1)
        old = "        from retained_qualification import qualification\n        reports = qualification(base)"
        new = '''        from fresh_qualification import qualification
        for mode in ('normal', 'disabled'):
            prefix = [] if mode == 'normal' else ['/usr/bin/env', 'DOTNET_EnableHWIntrinsic=0']
            worker('caller-' + mode, [*prefix, DOTNET, base/'caller/Caller.dll', base/'runtimes/production',
                   base/'caller/shapes.json', campaign/('caller-' + mode + '.json'), mode,
                   pin(base/'runtimes/portable/Lokad.Onnx.dll')['sha256']])
        for role in ROLES:
            manifest = base/'manifests'/(role+'-pyannote.json')
            worker(role+'-pyannote', [DOTNET, base/'runtimes'/role/'GraphQualification.dll',
                   base/'assets', manifest, campaign/(role+'-pyannote-output'),
                   pin(base/'runtimes'/role/'Lokad.Onnx.dll')['sha256']])
            worker(role+'-parakeet', [DOTNET, base/'runtimes'/role/'TranscribeReplay.dll',
                   '/home/vermorel/Onnx/models/parakeet-tdt-0.6b-v3',
                   base/'parakeet-reference/manifest.json', campaign/(role+'-parakeet.json')])
        reports = qualification(base, campaign)'''
        text = replace(text, old, new, 1)
    if name == 'audit_results.py':
        old = "    expected += ['native-conformance', 'meetings-inputs', 'meetings-run']"
        new = "    expected += ['caller-normal', 'caller-disabled'] + [r+'-'+f for r in ROLES for f in ['pyannote', 'parakeet']]\n" + old
        text = replace(text, old, new, 1)
        text = replace(text, "3290 if suite == 'backend' else 342", "3311 if suite == 'backend' else 343", 1)
        text = replace(text, "    from retained_qualification import qualification\n    reports = qualification(payload)",
                       "    from fresh_qualification import qualification\n    reports = qualification(payload, campaign)", 1)
        text = replace(text, 'retained_arrays_pyannote=36, retained_arrays_parakeet=1568, retained_pyannote_public_calls=32',
                       'fresh_arrays_pyannote=36, fresh_arrays_parakeet=1568, fresh_pyannote_public_calls=32', 1)
    target.write_text(text, encoding='utf8')
print('Derived', len(NAMES), 'campaign tools; unchanged resource, numerical and performance gates.')
