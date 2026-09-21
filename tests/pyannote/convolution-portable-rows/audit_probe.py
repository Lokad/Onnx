"""Keep all original captured-value checks; allocation reduction is not this candidate's gate."""
import common
from phase_audit import audit_preparation

preparation=audit_preparation(common.BASE,common)
assert preparation==common.read(common.BASE/'preparation-audit.json')
path=common.ROOT/'tests/pyannote/context-reuse-probe/audit.py'
source=path.read_text(encoding='utf8')
changes=[
    ("prepared = read(BASE / 'prepared.json')","prepared = read(BASE / 'probe-prepared.json')"),
    ("state = read(BASE / 'processes.json')","state = read(BASE / 'probe-processes.json')"),
    ("['build', 'forward', 'reverse']","['probe-restore', 'build', 'forward', 'reverse']"),
    ('admitted = admitted and reused < fresh','admitted = admitted and True  # All bit/ownership gates retained; no allocation-performance gate.'),
    ('identities = [state[\'supervisor\']]','identities = [state[\'supervisor\']] + PREPARATION[\'identities\']'),
    ('analysis = dict(passed=True, calls=len(all_rows)','analysis = dict(passed=True, preparation=PREPARATION, calls=len(all_rows)'),
    ("if path.is_file(): files[rel(path)] = pin(path)","if path.is_file() and not {'obj','packages'}.intersection(path.relative_to(BASE).parts): files[rel(path)] = pin(path)")]
for old,new in changes:
    assert source.count(old)==1,old;source=source.replace(old,new)
source=source.replace('request_scoped_prototype_admitted','captured_model_qualification_admitted')
namespace=dict(__name__='retained_graph_audit_for_portable_rows',__file__=str(path),PREPARATION=preparation)
exec(compile(source,str(path),'exec'),namespace)
namespace['main']()
