"""Finish the unlaunched bundle after preserving a line-ending-only refusal."""
from pathlib import Path
import sys

original = Path(__file__).with_name('prepare.py')
source = original.read_text(encoding='utf8')
start = source.index('    assert not PREPARED.exists() and not BASE.exists()')
end = source.index('    spec = dict(old)', start)
prefix = '''    assert PREPARED.exists() and BASE.exists() and not (BASE / 'prepared.json').exists()
    assert not (PREPARED / 'prepared.json').exists() and not (BASE / 'execution').exists()
    assert pin(CONSUMERS / 'closed.json')['sha256'] == '602d752956d6cf830851cd0dc35eb4bdf1230880fb2dce6d8bbea6f530bb555c'
    for name, wanted in read(CONSUMERS / 'closed.json')['files'].items():
        assert pin(ROOT / name) == wanted
    verified_files(PRIOR, read(PRIOR / 'failure-closed.json')['files'])
    payload = PREPARED / 'payload'
    old = read(PRIOR_PAYLOAD / 'payload.json')
    verified_files(PRIOR_PAYLOAD, old['files'])
    changes = {}
    for name, wanted in old['files'].items():
        current = pin(payload / name)
        if current != wanted:
            assert name.startswith(('runtimes/portable/GraphQualification.', 'runtimes/rows/GraphQualification.'))
            changes[name] = dict(before=wanted, after=current)
    assert {'runtimes/portable/GraphQualification.dll', 'runtimes/rows/GraphQualification.dll'} <= set(changes)
    retained(payload)
    verified_files(payload, read(PRIOR / 'collected/campaign/built-files.json'))
    admission = TOOLS / 'admission.py'
    previous = ROOT / 'tests/pyannote/combined-amd/admission.py'
    assert admission.read_text(encoding='utf8') == previous.read_text(encoding='utf8')
    assert pin(previous) == old['protocol']['performance_admission']
    assert pin(admission) != pin(previous)
    copy(admission, BASE / 'refused-admission-copy.py')
    write(BASE / 'preparation-refusal.json', dict(passed=False, vm_deployed=False,
        cause='Generated admission copy changed only line endings; strict pre-freeze byte guard refused.',
        before=pin(admission), original=pin(previous), retained_source=pin(BASE / 'refused-admission-copy.py')))
    # This module has not run or been frozen in an execution bundle. Preserve
    # its refused bytes above and restore the exact prospective policy bytes.
    copy(previous, admission)
'''
source = source[:start] + prefix + source[end:]
namespace = dict(__name__='complete_preparation', __file__=str(original))
exec(compile(source, str(original) + ' [unfinished preparation]', 'exec'), namespace)

if __name__ == '__main__':
    namespace['main']()
