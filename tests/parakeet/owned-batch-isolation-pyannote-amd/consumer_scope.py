"""Preserve numerical checks and monitoring while removing redundant build jobs."""
from pathlib import Path
from protocol import pin

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
ORIGINAL = TOOLS.parent/'packed-final-row-pyannote-amd'


def expected(name):
    value = (ORIGINAL/name).read_text(encoding='utf8')
    if name == 'protocol.py':
        value = value.replace('Frozen resource and input contracts for eight actual-product numerical jobs.',
                              'Frozen contracts for two actual-product Pyannote correctness jobs.')
        value = value.replace("JOBS = ['consumer-restore','consumer-build','consumer-inventory','selected','candidate']",
                              "JOBS = ['selected','candidate']")
    if name == 'remote.py':
        value = value.replace('Build the literal-only consumer, then qualify both complete measured Pyannote products.',
                              'Reuse qualified consumers and test both actual Pyannote products.')
        start, end = value.index("FLAGS=['--tl:off'"), value.index('\ndef main():')
        value = value[:start]+'''def command_for(name,spec):
    assert name in ['selected','candidate']
    return [DOTNET,BASE/'runtimes'/name/'GraphQualification.dll',BASE/'assets',BASE/'manifests'/(name+'-pyannote.json'),BASE/name/'output',spec['identities'][name]['Lokad.Onnx.dll']['sha256']],False


def after(name,spec,row):
    assert name in ['selected','candidate']
    save(BASE/name/'review.json',model(BASE,name))

'''+value[end:]
        value = value.replace("    for name in ['logs','tmp','packages','cli-home','http-cache','empty-feed','built']:(BASE/name).mkdir()",
                              "    from reuse_checks import verify_reuse\n    verify_reuse(BASE, spec)\n    for name in ['logs','tmp']:(BASE/name).mkdir()")
        value = '\n'.join(line for line in value.split('\n') if not line.startswith('    build_env=dict(env,'))
        value = value.replace('                        job_env=build_env if build else env',
                              '                        assert not build\n                        job_env=env')
        value = value.replace("            if (BASE/'built.json').exists():\n                for file,wanted in read(BASE/'built.json')['files'].items():assert pin(BASE/file)==wanted,file\n", '')
    if name == 'audit.py':
        value = value.replace('from prepare import APP_PAYLOAD,MODEL,OLD_DATA,NEW_DATA',
                              'from prepare import APP_PAYLOAD,MODEL\nfrom reuse_checks import verify_reuse')
        value = value.replace('M78 qualified release Coref95a13c5/Dataa893952f', 'Qualified release Coref95a13c5/Dataa893952f')
        value = value.replace('M78 packed final row Core49901366/Data01e9e784', 'Dispatch relocation Coree07a4518/Data01e9e784')
        start = value.index("    assert (collected/'evidence/original-consumer.cs')")
        end = value.index('    original_payload=', start)
        value = value[:start]+value[end:]
        start = value.index("    built=read(collected/'built.json')")
        end = value.index('    results={}', start)
        value = value[:start]+"    il = verify_reuse(collected, payload)\n"+value[end:]
        value = value.replace("consumers=dict(payload['consumers'],candidate=built['consumer'])", "consumers=payload['consumers']")
        value = value.replace('inventory=il,results=results', 'retained_consumer_inventory=il,consumer_rebuilt=False,results=results')
    if name == 'run.py':
        value = value.replace('packed-final-row-pyannote', 'owned-batch-isolation-pyannote')
        value = value.replace("[*JOBS,'logs','built','evidence','manifests','graph-reference']",
                              "[*JOBS,'logs','runtimes','evidence','manifests','graph-reference']")
        value = value.replace('def observe():\n', "def observe():\n    assert not (BASE/'closed.json').exists(), 'Preserve the closed campaign'\n")
    return value


def verify_scope():
    files = {}
    exact = ['checks.py', 'candidate_protocol.py', 'qualify_outputs.py', 'test_semantics.py']
    for name in [*exact, 'protocol.py', 'remote.py', 'audit.py', 'run.py']:
        original = ORIGINAL/name
        if name in exact:
            assert (TOOLS/name).read_bytes() == original.read_bytes(), name
        else:
            assert (TOOLS/name).read_text(encoding='utf8') == expected(name), name
        files[original.relative_to(ROOT).as_posix()] = pin(original)
    return files


if __name__ == '__main__':
    print(verify_scope())
