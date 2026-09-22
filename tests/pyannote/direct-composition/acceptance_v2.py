"""Correct only the caller schedule count: 22 + 336 + 36 + 6 = 400."""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
source = (TOOLS / 'acceptance.py').read_text(encoding='utf8')
assert source.count('pyannote-direct-composition-acceptance-20260922') == 2
source = source.replace('pyannote-direct-composition-acceptance-20260922',
                        'pyannote-direct-composition-acceptance-v2-20260922')
old = "    text=(TOOLS / 'CallerProbe.cs').read_text(encoding='utf8')"
assert source.count(old) == 1
source = source.replace(old, old + '''
    assert text.count('Require(records.Count == 736, "Complete schedule");') == 1
    text = text.replace('Require(records.Count == 736, "Complete schedule");',
                        'Require(records.Count == 400, $"Complete schedule: {records.Count}");')''')
# Python assertions and the emitted preparation manifest use the same actual count.
source = source.replace("len(result['records'])==736", "len(result['records'])==400")
source = source.replace('caller_cases_per_mode=736', 'caller_cases_per_mode=400')
old = '    assert not BASE.exists()'
assert source.count(old) == 1
source = source.replace(old, old + '''
    previous = ROOT / 'artifacts/pyannote-direct-composition-acceptance-20260922/failure-closed.json'
    failed = read(previous)
    assert failed['passed'] and failed['expected_failure']
    verify(failed['files'])
    for identity in failed['identities']: terminal(identity)''')
namespace = dict(__name__='caller_count_successor', __file__=str(TOOLS / 'acceptance.py'))
exec(compile(source, str(TOOLS / 'acceptance.py'), 'exec'), namespace)
configured = namespace['configured']

if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'qualify']
    if sys.argv[1] == 'prepare':
        namespace['prepare']()
    else:
        c = configured()
        code = (TOOLS / 'qualify.py').read_text(encoding='utf8').replace('from prepare import *', '')
        exec(compile(code, str(TOOLS / 'qualify.py'), 'exec'), c)
        c['main']()
