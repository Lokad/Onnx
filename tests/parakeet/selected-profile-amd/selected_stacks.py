"""Accept the exporter's optional thread name without dropping any profile."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT/'artifacts/pyannote-selected-profile-amd-20260922/payload/tools/stacks.py'
source = ORIGINAL.read_text(encoding='utf8')
changes = [
    ("document['$schema'].endswith('speedscope/file-format-schema.json')",
     "document['$schema'] == 'https://www.speedscope.app/file-format-schema.json'"),
    ("assert re.fullmatch(r'Thread \\(\\d+\\)', profile['name'])",
     "assert re.fullmatch(r'Thread \\(\\d+\\)(?: \\([^\\r\\n]+\\))?', profile['name'])"),
]
for old,new in changes:
    assert source.count(old) == 1,old; source = source.replace(old,new)
namespace = dict(__name__='retained_stack_parser',__file__=str(ORIGINAL))
exec(compile(source,str(ORIGINAL),'exec'),namespace)


def identities(document):
    rows = []
    for profile in document['profiles']:
        match = re.fullmatch(r'Thread \((\d+)\)(?: \([^\r\n]+\))?',profile['name'])
        assert match is not None,profile['name']
        rows.append(int(match.group(1)))
    assert len(set(rows)) == len(rows),'Duplicate exported native thread ID'


def inspect(document,markers):
    identities(document)
    return namespace['inspect'](document,markers)


def cross_export(speedscope,chromium):
    identities(speedscope)
    return namespace['cross_export'](speedscope,chromium)
