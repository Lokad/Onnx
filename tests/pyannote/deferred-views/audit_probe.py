"""Preserve every original captured-output and resource check."""
from common import *

closure = read(BASE / 'focused-closed.json')
assert closure['passed']
verify(closure['files'])
for identity in closure['identities']:
    terminal(identity)
preparation = read(BASE / 'focused-analysis.json')
assert preparation == read(BASE / 'preparation-audit.json')
path = ROOT / 'tests/pyannote/context-reuse-probe/audit.py'
source = path.read_text(encoding='utf8')
changes = [
    ("prepared = read(BASE / 'prepared.json')", "prepared = read(BASE / 'probe-prepared.json')"),
    ("state = read(BASE / 'processes.json')", "state = read(BASE / 'probe-processes.json')"),
    ("['build', 'forward', 'reverse']", "['probe-restore', 'build', 'forward', 'reverse']"),
    ('admitted = admitted and reused < fresh', 'admitted = admitted and True  # Allocation counters retained, not a numerical qualification gate.'),
    ("identities = [state['supervisor']]", "identities = [state['supervisor']] + PREPARATION['identities']"),
    ('analysis = dict(passed=True, calls=len(all_rows)', 'analysis = dict(passed=True, preparation=PREPARATION, calls=len(all_rows)'),
    ('if path.is_file(): files[rel(path)] = pin(path)', "if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(BASE).parts): files[rel(path)] = pin(path)")]
for before, after in changes:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
source = source.replace('request_scoped_prototype_admitted', 'captured_model_qualification_admitted')
namespace = dict(__name__='original_deferred_views_graph_audit', __file__=str(path), PREPARATION=preparation)
exec(compile(source, str(path), 'exec'), namespace)

if __name__ == '__main__':
    namespace['main']()
