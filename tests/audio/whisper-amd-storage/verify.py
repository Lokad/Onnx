"""Verify closed files and every printed row from raw integer timer ticks."""
import math
import re
from common import *
from statistics_exact import exact_rows, suffix


def checked_rows():
    base = BASE/'collected'; state = read(base/'campaign/identity.json'); audit = read(BASE/'audit.json')
    assert state['complete'] and state['code'] == 0 and audit['passed']
    workers = [read(base/r['output']/'worker/result.json') for r in state['runs']]
    cases = read(base/'manifests/whisper.json')['cases']; assert len(cases) == 20
    rows = exact_rows(workers,cases); assert len(rows) == len(audit['table']) == 21
    checks = 0
    for expected, observed in zip(rows,audit['table'],strict=True):
        assert expected['name'] == observed['name']
        for engine in ['managed','ort']:
            for key in ['seconds','rtf']:
                assert math.isclose(float(expected[engine][key]),observed[engine][key],rel_tol=1e-14); checks += 1
            for a,b in zip(expected[engine]['visits'],observed[engine]['visits'],strict=True):
                for x,y in zip(a['passes'],b['passes'],strict=True):
                    assert math.isclose(float(x),y,rel_tol=1e-14); checks += 1
                assert math.isclose(float(a['mean']),b['mean'],rel_tol=1e-14); checks += 1
        assert math.isclose(float(expected['ratio']),observed['ratio'],rel_tol=1e-14); checks += 1
    return rows, checks


def main():
    closed = read(BASE/'closed.json'); assert closed['passed']
    for name, expected in closed['files'].items():
        assert pin(ROOT/name) == expected, name
    rows, checks = checked_rows()
    report = TOOLS/'results.md'; body = report.read_text(encoding='utf8')
    benchmark = (BASE/'document-snapshots/BENCHMARK.md').read_text(encoding='utf8')
    section = benchmark.split('### Audio: matched AMD Whisper baseline\n',1)[1].split('\n### ',1)[0]
    for row in rows:
        assert suffix(row) in body
    assert suffix(rows[0]) in section
    links = 0
    for path in [report,ROOT/'BENCHMARK.md',ROOT/'docs/model-support.md']:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf8')):
            if '://' in link or link.startswith('#'):
                continue
            assert (path.parent/link.split('#')[0]).exists(),(path,link); links += 1
    ssh(PRELUDE+'terminal(%r)\n' % closed['births'])
    write(BASE/'final-verification.json',dict(passed=True,closure=pin(BASE/'closed.json'),files=len(closed['files']),
        decimal_checks=checks,table_rows=len(rows),links=links,births=closed['births']))
    print(json.dumps(read(BASE/'final-verification.json')))


if __name__=='__main__':
    main()
