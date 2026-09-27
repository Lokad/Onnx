"""Reuse all raw-clock checks; tighten repeatability and declare fallback controls."""
from pathlib import Path
import types
from protocol import PREPARED_JOBS, TIMING_JOBS

source = Path(__file__).with_name('checks_base.py')
if not source.exists(): source = source.parent.parent/'prepared-recurrence-timing-amd/checks.py'
text = source.read_text()
changes = {
    '(13107200 if role==\'candidate\' else 0)': "(0 if role.endswith('fallback') else 13107200)",
    'def evaluate(workers,cases):': 'def evaluate(workers,cases,corpus_limit=90):',
    "110 if group=='corpus' else 120": '110',
    "90 if group=='corpus' else 105": "corpus_limit if group=='corpus' else 105",
}
for before, after in changes.items():
    assert text.count(before) == (2 if before.startswith('110') else 1), before
    text = text.replace(before, after)
inherited = types.ModuleType('layout_timing_checks')
exec(compile(text, str(source), 'exec'), inherited.__dict__)
inherited.TIMING_JOBS = PREPARED_JOBS
qualify = inherited.qualify


def evaluate(workers, cases):
    assert set(workers) == set(TIMING_JOBS)
    reports = {}
    for path, suffix, limit in [('prepared', '', 90), ('fallback', 'fallback', 105)]:
        selected = {n.replace(suffix, '') if suffix else n: value for n, value in workers.items()
            if ('fallback' in n) == bool(suffix)}
        reports[path] = inherited.evaluate(selected, cases, limit)
    controls = [dict(path=path, **row) for path, report in reports.items() for row in report['controls']]
    gates = [dict(path=path, **row) for path, report in reports.items() for row in report['gates']]
    table = [dict(path=path, **row) for path, report in reports.items() for row in report['table']]
    assert len(controls) == 168 and len(gates) == 28
    return dict(controls_passed=all(c['passed'] for c in controls),
        admitted=all(c['passed'] for c in [*controls, *gates]), controls=controls, gates=gates, table=table,
        policy='One fixed layout; complete captured LSTMs; five warmup/five measured passes; all repeatability <=1.10; prepared corpus gain>=10%, no prepared case>5% slower; every unchanged fallback ratio<=1.05; all clocks retained.')
