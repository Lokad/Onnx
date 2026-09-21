"""Add an explicit correction to the generated report's skipped-test column."""
import collections
import xml.etree.ElementTree as ET
from common import *

OUTPUT = ROOT / 'artifacts/pyannote-trx-skips-20260922'


def main():
    assert not OUTPUT.exists()
    closure_path = BASE / 'closed.json'
    assert pin(closure_path)['sha256'] == 'f687396b6a6966b80e41f100dbe3bdddd0c682c8d63bfc658744ce05f5d79a3d'
    closure = read(closure_path)
    files, rows = {rel(closure_path): pin(closure_path), rel(Path(__file__)): pin(Path(__file__))}, []
    for name, passed, skipped in [('request-focused', 2, 0), ('backend-full', 3280, 93), ('tensors-full', 342, 0)]:
        path = BASE / 'test-results' / (name + '.trx')
        assert closure['files'][rel(path)] == pin(path)
        tree = ET.parse(path)
        counters = tree.find('.//{*}Counters').attrib
        outcomes = collections.Counter(r.attrib['outcome'] for r in tree.findall('.//{*}UnitTestResult'))
        assert outcomes['Passed'] == passed and outcomes['NotExecuted'] == skipped and not outcomes['Failed']
        assert int(counters['total']) == passed + skipped and int(counters['passed']) == int(counters['executed']) == passed
        files[rel(path)] = pin(path)
        rows.append(dict(name=name, passed=passed, retained_skips=skipped, raw_counters=counters, individual_outcomes=dict(outcomes)))
    original = TOOLS / 'results-20260921.md'
    assert '| backend-full | 3280 | 0 |' in original.read_text(encoding='utf8')
    files[rel(original)] = pin(original)
    correction = TOOLS / 'trx-skip-correction-20260922.md'
    assert not correction.exists()
    correction.write_text('''# Correction: the backend report retained 93 skipped tests

The September21 sparse-mel application report's backend skipped-test column
incorrectly says zero. The correct result is **3,280 passed, 93 skipped, zero
failed**. Request-focused remains2passed/0skipped and tensors342passed/0skipped.

The original TRX has3,373individual outcomes:3,280Passed and93NotExecuted.
Its aggregate total/executed/passed counters are3373/3280/3280, but the aggregate
notExecuted field is0. The exporter used that aggregate field for the skipped
column. The independent application auditor already checked all individual
outcomes and required93skips; its qualification verdict is unchanged.

The original report, exporter, TRX and closed evidence remain preserved. This
correction changes no tests, numerical bound, public result, timing sample or
benchmark ratio. Future exporters count skipped individual records and retain
the raw aggregate counters separately.

Evidence is artifacts/pyannote-trx-skips-20260922/closed.json. Run
verify_skip_counts.py using C:/Python313/python.exe -X utf8 -B to reproduce in
a fresh output location; the existing correction is immutable.
''', encoding='utf8')
    OUTPUT.mkdir()
    save(OUTPUT / 'analysis.json', dict(passed=True, rows=rows, original_report=pin(original), correction=pin(correction),
        scope='Reporting correction only; original individual-outcome qualification and timing verdicts unchanged.'))
    files[rel(correction)] = pin(correction)
    files[rel(OUTPUT / 'analysis.json')] = pin(OUTPUT / 'analysis.json')
    verify(files)
    save(OUTPUT / 'closed.json', dict(passed=True, files=files, analysis=pin(OUTPUT / 'analysis.json')))
    print(json.dumps(dict(closed=pin(OUTPUT / 'closed.json'), backend_passed=3280, retained_skips=93)))


if __name__ == '__main__':
    main()
