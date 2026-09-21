"""Keep every original bit/counter check and add comparison with the predecessor."""
import common
from phase_audit import audit_preparation

common.BASE = common.ROOT / 'artifacts/pyannote-convolution-pool-v4-20260921'
preparation = audit_preparation(common.BASE, common)
assert preparation == common.read(common.BASE / 'preparation-audit.json')
source_path = common.ROOT / 'tests/pyannote/context-reuse-probe/audit.py'
source = source_path.read_text(encoding='utf8')


def replace(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)


replace("prepared = read(BASE / 'prepared.json')", "prepared = read(BASE / 'probe-prepared.json')")
replace("state = read(BASE / 'processes.json')", "state = read(BASE / 'probe-processes.json')")
replace("['build', 'forward', 'reverse']", "['probe-restore', 'build', 'forward', 'reverse']")
replace("    analysis = dict(passed=True, calls=len(all_rows)", """    comparisons = []
    previous = read(PROBE / 'analysis.json')
    for order in ['forward', 'reverse']:
        before = next(s for s in previous['summaries'] if s['order'] == order and s['graph'] == 'embedding' and s['mode'] == 'reuse' and s['phase'] == 'repeat')
        after = next(s for s in summaries if s['order'] == order and s['graph'] == 'embedding' and s['mode'] == 'reuse' and s['phase'] == 'repeat')
        ratio = after['allocated_mean'] / before['allocated_mean']
        comparisons.append(dict(order=order, predecessor=before, candidate=after, allocation_ratio=ratio))
        admitted = admitted and ratio < 1
    analysis = dict(passed=True, predecessor_comparisons=comparisons, preparation=PREPARATION, calls=len(all_rows)""")
source = source.replace('request_scoped_prototype_admitted', 'application_qualification_admitted')
replace("        if path.is_file(): files[rel(path)] = pin(path)",
    "        if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(BASE).parts): files[rel(path)] = pin(path)")
namespace = dict(__name__='original_graph_auditor_with_predecessor_comparison', __file__=str(source_path), PREPARATION=preparation)
exec(compile(source, str(source_path), 'exec'), namespace)
namespace['main']()
