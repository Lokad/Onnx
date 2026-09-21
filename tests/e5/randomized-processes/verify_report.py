"""Independently recompute complete e5 report intervals from integer raw ticks."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
import re

from contract import pin, read, write
from evidence import verify_inventory
from remote import BASE, ROOT


def decimal_interval(y, x, critical):
    with localcontext() as ctx:
        ctx.prec = 80
        def d(value):
            return Decimal(value.numerator)/Decimal(value.denominator)
        x, y = [d(v) for v in x], [d(v) for v in y]
        n = Decimal(len(x)); sx, sy = sum(x), sum(y)
        mx, my = sx/n, sy/n
        vx = (sum(v*v for v in x)-sx*sx/n)/(n*(n-1))
        vy = (sum(v*v for v in y)-sy*sy/n)/(n*(n-1))
        covariance = (sum(a*b for a, b in zip(x, y, strict=True))-sx*sy/n)/(n*(n-1))
        q = Decimal.from_float(critical)**2
        a, b, c = mx*mx-q*vx, mx*my-q*covariance, my*my-q*vy
        if a <= 0:
            return dict(bounded=False, interval=None, ratio=float(my/mx))
        discriminant = b*b-a*c
        # Only roundoff at80decimal digits may be clipped. This also permits
        # exact proportional fixtures whose mathematical discriminant is zero.
        if discriminant < 0:
            assert abs(discriminant) <= max(abs(b*b), abs(a*c))*Decimal('1e-65')
            discriminant = Decimal(0)
        radius = discriminant.sqrt()
        return dict(bounded=True, interval=[float((b-radius)/a), float((b+radius)/a)], ratio=float(my/mx))


def verify_statistics(base, phase, analysis):
    meta = read(base/'frozen.json'); jobs = meta['schedules'][phase]
    assert len(jobs) == 1800
    process = {}; measured_count = conditioning_count = 0
    for job in jobs:
        value = read(base/('result-'+phase)/job['name']/'output/result.json')
        expected_calls = [32, 16, 4, 4, 2][job['case_index']]
        rows = value['measured']; freq = value['frequency']
        assert type(freq) is int and freq > 0 and len(rows) == 16*expected_calls
        for i, row in enumerate(rows):
            assert (row['block'], row['call']) == divmod(i, expected_calls)
            assert type(row['execute']) is int and type(row['request']) is int and 0 < row['execute'] <= row['request']
        key = (job['case'], job['policy'], job['role'], job['cohort'])
        assert key not in process
        process[key] = {b: Fraction(sum(r[b] for r in rows), freq*len(rows)) for b in ('execute', 'request')}
        measured_count += len(rows); conditioning_count += len(value['conditioning'])
    assert measured_count == analysis['measured_calls'] == 334080
    assert conditioning_count == analysis['conditioning_calls']
    critical = analysis['timing']['critical']; contrasts = 20 if phase == 'aa' else 60
    assert analysis['timing']['contrast_count'] == contrasts
    # Independent Simpson integration of the Student59density verifies the
    # critical value without importing the production quantile implementation.
    df, steps = 59, 20000
    factor = math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)
    density = lambda t: factor*(1+t*t/df)**(-(df+1)/2)
    step = critical/steps
    central = 2*step/3*(density(0)+density(critical)+sum((4 if i % 2 else 2)*density(i*step) for i in range(1, steps)))
    assert abs(central-(1-.05/contrasts)) < 2e-12
    cases = ['e5-8tok', 'e5-30tok', 'e5-30pad128', 'e5-128tok', 'e5-512tok']
    expected_rows = {(c, p, b) for c in cases for p in ('default', 'memory') for b in ('execute', 'request')}
    assert {(r['case'], r['policy'], r['boundary']) for r in analysis['timing']['results']} == expected_rows
    assert len(analysis['timing']['results']) == len(expected_rows)
    expected_summaries = {(c, p, r) for c in cases for p in ('default', 'memory') for r in ('A', 'C', 'N')}
    assert {(r['case'], r['policy'], r['role']) for r in analysis['summaries']} == expected_summaries
    assert len(analysis['summaries']) == len(expected_summaries)
    screens, diagnostic, checked = [], [], []
    for row in analysis['timing']['results']:
        assert set(row['contrasts']) == ({'C/A'} if phase == 'aa' else {'C/A', 'C/N', 'A/N'})
        values = {role: [process[(row['case'], row['policy'], role, c)][row['boundary']] for c in range(60)] for role in ('A', 'C', 'N')}
        for label, reported in row['contrasts'].items():
            numerator, denominator = label.split('/')
            y, x = values[numerator], values[denominator]
            actual = decimal_interval(y, x, critical)
            assert actual['bounded'] is reported['bounded'] and abs(actual['ratio']-reported['ratio']) < 2e-12
            assert reported['n'] == 60
            assert reported['process_means'] == dict(numerator=[float(v) for v in y], denominator=[float(v) for v in x])
            if actual['bounded']:
                assert max(abs(a-b) for a, b in zip(actual['interval'], reported['interval'], strict=True)) < 2e-12
            else:
                assert reported['interval'] is None
            ratio = sum(y)/sum(x)
            differences = [a-ratio*b for a, b in zip(y, x, strict=True)]
            mean = sum(differences)/60
            squares = [(v-mean)**2 for v in differences]
            share = float(max(squares)/sum(squares)) if sum(squares) else 0.
            assert abs(share-reported['largest_observed_variance_share']) < 2e-12
            diagnostic.append(share <= .2)
            checked.append(actual)
        control = row['contrasts']['C/A']; bounds = control['interval']
        if phase == 'aa':
            passed = control['bounded'] and .99 <= bounds[0] <= 1 <= bounds[1] <= 1.01
        else:
            case_index = ['e5-8tok', 'e5-30tok', 'e5-30pad128', 'e5-128tok', 'e5-512tok'].index(row['case'])
            passed = control['bounded'] and 0 <= bounds[0] <= bounds[1] <= [.98, .99, 1.01, 1.01, 1.01][case_index]
        assert passed is row['statistical_screen']; screens.append(passed)
    assert len(checked) == contrasts and len(screens) == 20
    assert all(screens) is analysis['timing']['statistical_screen']
    assert all(diagnostic) is analysis['diagnostic_screen']
    for row in analysis['summaries']:
        for boundary in ('execute', 'request'):
            means = [process[(row['case'], row['policy'], row['role'], c)][boundary] for c in range(60)]
            assert abs(float(sum(means)/60)*1000-row['boundaries'][boundary]['mean_ms']) < 1e-9
    return dict(intervals=contrasts, measured_calls=measured_count, conditioning_calls=conditioning_count,
                statistical_screen=all(screens), diagnostic_screen=all(diagnostic))


def verify_markdown(text, analysis):
    summaries = {(r['case'], r['policy'], r['role']): r for r in analysis['summaries']}
    tables = [line for line in text.splitlines() if line.startswith('| e5-')]
    rows = [r for boundary in ('execute', 'request') for r in analysis['timing']['results'] if r['boundary'] == boundary]
    assert len(tables) == len(rows) == 20
    for line, row in zip(tables, rows, strict=True):
        cells = [v.strip() for v in line.strip('|').split('|')]
        assert len(cells) == 7 and cells[:2] == [row['case'], row['policy']]
        means = {role: summaries[(row['case'], row['policy'], role)]['boundaries'][row['boundary']]['mean_ms'] for role in ('A', 'C', 'N')}
        for cell, role in zip(cells[2:5], ('A', 'C', 'N'), strict=True):
            assert abs(float(cell)-means[role]) <= .000000500001
        labels = ['C/A'] if analysis['phase'] == 'aa' else ['C/A', 'C/N']
        for cell, label in zip(cells[5:5+len(labels)], labels, strict=True):
            wanted = row['contrasts'][label]
            if not wanted['bounded']:
                assert cell == 'unbounded'
            else:
                parsed = re.fullmatch(r'(-?[0-9.]+) \[(-?[0-9.]+), (-?[0-9.]+)\]', cell)
                assert parsed
                for a, b in zip(map(float, parsed.groups()), [wanted['ratio']]+wanted['interval'], strict=True):
                    assert abs(a-b) <= .00005000001
        if analysis['phase'] == 'aa':
            assert abs(float(cells[6])-means['A']/means['N']) <= .00005000001
    return len(tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['aa', 'compare'], required=True)
    args = parser.parse_args(); phase = args.phase
    target = BASE/(phase+'-verification.json'); assert not target.exists()
    base = BASE/('collected-'+phase); meta, collection = verify_inventory(base, phase)
    analysis = read(BASE/(phase+'-analysis.json')); record = read(BASE/(phase+'-report.json'))
    folder = ROOT/'tests/e5/randomized-processes'; report = folder/(phase+'-results.md'); observations = folder/(phase+'-observations.json')
    assert pin(BASE/(phase+'-analysis.json')) == record['analysis'] and pin(report) == record['report'] and pin(observations) == record['observations']
    assert read(observations) == analysis
    result = verify_statistics(base, phase, analysis)
    result['markdown_rows'] = verify_markdown(report.read_text(), analysis)
    result.update(passed=True, frozen=pin(base/'frozen.json'), collection=pin(base/(phase+'-collection.json')), report=pin(report), observations=pin(observations))
    write(target, result)
    if phase == 'aa' and result['statistical_screen'] and result['diagnostic_screen']:
        retained = dict(collection['files']); retained['aa-collection.json'] = pin(base/'aa-collection.json')
        gate = dict(passed=True, phase='aa', frozen=pin(base/'frozen.json'), identity=analysis['identity'],
                    timing=dict(statistical_screen=True), diagnostic_screen=True, resources=dict(births=collection['births']),
                    retained_files=retained, independent_verification=pin(target), report=pin(report),
                    assumptions='Approximate finite-campaign inference; no interference and regularity remain assumptions.')
        write(BASE/'aa-gate.json', gate)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
