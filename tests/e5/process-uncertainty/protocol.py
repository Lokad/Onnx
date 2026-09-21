"""Prospective fixed fresh-process design. No VM runner is implied by this file."""
from fractions import Fraction
from hashlib import sha256
from itertools import permutations
from functools import lru_cache

from estimator import fieller, contained, student_critical

PROTOCOL = 'e5-fresh-process-uncertainty-v1'
SEED = 'e5-process-uncertainty-20260921-fixed-before-new-inference'
CASES = ['e5-8tok', 'e5-30tok', 'e5-30pad128', 'e5-128tok', 'e5-512tok']
POLICIES = ['default', 'memory']
ROLES = ['A', 'C', 'N']
COHORTS = 60
CORE = 'd1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4'
NATIVE = '13ab8084954fa4a47c777880180b90810d6020f021441395712b48a75b74c68b'
CRITERIA = dict(family_alpha=.05, aa_equivalence=[.99, 1.01],
                candidate_upper=[.98, .99, 1.01, 1.01, 1.01], primary_parity=1.05,
                native_scaled_error=1e-4, foreign_cpu=.02, steal=.005)
LIMITS = dict(seconds=300, rss=6 * 1024**3, available=1024**3)


def _key(*parts):
    return sha256('|'.join([SEED] + [str(v) for v in parts]).encode('ascii')).digest()


def schedule(phase):
    if phase not in ('aa', 'compare'):
        raise ValueError('unknown phase')
    result = []
    for cohort in range(COHORTS):
        cases = sorted([(i, p) for i in range(5) for p in POLICIES], key=lambda item: _key(phase, cohort, *item))
        for index, policy in cases:
            orders = sorted(permutations(ROLES), key=lambda order: _key(phase, cohort // 6, index, policy, ''.join(order)))
            order = orders[cohort % 6]
            for position, role in enumerate(order):
                result.append(dict(name=f'c{cohort:02d}-{CASES[index]}-{policy}-{role}', cohort=cohort,
                                   case=CASES[index], case_index=index, policy=policy, role=role, position=position))
    return result


def specification(job, phase, smoke=False):
    if phase not in ('aa', 'compare'):
        raise ValueError('unknown phase')
    return dict(protocol=PROTOCOL, phase=phase, policy=job['policy'], role=job['role'],
                cohort=job['cohort'], case_index=job['case_index'], blocks=2 if smoke else 16,
                calls=2 if smoke else [32, 16, 4, 4, 2][job['case_index']],
                conditioning_seconds=.1 if smoke else 30, conditioning_minimum=2 if smoke else 128, smoke=smoke)


@lru_cache(maxsize=2)
def critical(phase):
    if phase not in ('aa', 'compare'):
        raise ValueError('unknown phase')
    contrasts = 20 if phase == 'aa' else 60
    return student_critical(1 - CRITERIA['family_alpha'] / contrasts, COHORTS - 1)


def evaluate(means, phase):
    """Evaluate exactly one complete schedule of process means.

    This is the statistical layer ONLY. A runner must separately validate raw
    calls, binaries, actual processes, resources, assumptions and matching A/A.
    Each input is {job: exact scheduled identity, execute: Fraction seconds,
    request: Fraction seconds}; ordering, omissions and duplicate jobs refuse.
    """
    jobs = schedule(phase)
    if len(means) != len(jobs):
        raise ValueError('incomplete process schedule')
    for row, job in zip(means, jobs, strict=True):
        if row.get('job') != job:
            raise ValueError('changed, reordered or duplicate process identity')
        for boundary in ('execute', 'request'):
            if type(row.get(boundary)) not in (int, Fraction) or row[boundary] <= 0:
                raise ValueError('positive exact process mean required')
        if row['request'] < row['execute']:
            raise ValueError('request boundary shorter than Execute')
    results, all_pass = [], True
    contrasts = [('C', 'A')] if phase == 'aa' else [('C', 'A'), ('C', 'N'), ('A', 'N')]
    for index, case in enumerate(CASES):
        for policy in POLICIES:
            group = {(r['job']['cohort'], r['job']['role']): r for r in means
                     if r['job']['case'] == case and r['job']['policy'] == policy}
            for boundary in ('execute', 'request'):
                entries = {}
                for numerator, denominator in contrasts:
                    y = [group[(c, numerator)][boundary] for c in range(COHORTS)]
                    x = [group[(c, denominator)][boundary] for c in range(COHORTS)]
                    entries[numerator + '/' + denominator] = fieller(y, x, critical(phase))
                control = entries['C/A']
                if phase == 'aa':
                    passed = contained(control, *CRITERIA['aa_equivalence']) and control['interval'][0] <= 1 <= control['interval'][1]
                else:
                    passed = contained(control, 0, CRITERIA['candidate_upper'][index])
                all_pass &= passed
                parity = None if phase == 'aa' else contained(entries['C/N'], 0, CRITERIA['primary_parity'])
                results.append(dict(case=case, policy=policy, boundary=boundary, contrasts=entries,
                                    statistical_screen=passed, parity_bound=parity, primary=index < 4))
    return dict(protocol=PROTOCOL, phase=phase, cohorts=COHORTS, family_alpha=CRITERIA['family_alpha'],
                contrast_count=20 if phase == 'aa' else 60, statistical_screen=all_pass,
                assumptions_verified=False, ready_for_promotion=False, results=results)
