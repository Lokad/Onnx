"""Fixed-population inference from independently randomized three-role cohorts.

Potential outcomes can vary arbitrarily over positions. The assignment, not an
i.i.d. latency model, supplies independence. No-interference and large-sample
regularity remain assumptions; this module cannot prove them from observations.
"""
from fractions import Fraction as F
from itertools import permutations, product
from pathlib import Path
import importlib.util

ESTIMATOR = Path(__file__).resolve().parent.parent/'process-uncertainty/estimator.py'
_spec = importlib.util.spec_from_file_location('e5_process_fieller', ESTIMATOR)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
fieller = _module.fieller
student_critical = _module.student_critical
process_mean = _module.process_mean

PROTOCOL = 'e5-independent-assignment-v2'
ROLES = ('A', 'C', 'N')
ORDERS = tuple(permutations(ROLES))
CASES = ('e5-8tok', 'e5-30tok', 'e5-30pad128', 'e5-128tok', 'e5-512tok')
POLICIES = ('default', 'memory')
COHORTS = 60


def checked_population(population):
    rows = list(population)
    if len(rows) < 2:
        raise ValueError('at least two cohorts required')
    for table in rows:
        if len(table) != 3 or any(len(slot) != 3 for slot in table):
            raise ValueError('each cohort needs three positions and three role potentials')
        if any(type(v) not in (int, F) or v <= 0 for slot in table for v in slot):
            raise ValueError('positive exact potential durations required')
    return rows


def observed(table, order):
    if tuple(order) not in ORDERS:
        raise ValueError('exactly one of each role required')
    return {role: F(table[position][ROLES.index(role)]) for position, role in enumerate(order)}


def population_ratio(population, numerator='C', denominator='A'):
    """Ratio of means over ALL fixed execution positions, not a future population."""
    tables = checked_population(population)
    a, b = ROLES.index(numerator), ROLES.index(denominator)
    return F(sum(F(slot[a]) for table in tables for slot in table),
             sum(F(slot[b]) for table in tables for slot in table))


def potential_contrasts(table, ratio, numerator='C', denominator='A'):
    if type(ratio) not in (int, F) or ratio <= 0:
        raise ValueError('positive exact trial ratio required')
    return tuple(observed(table, order)[numerator] - ratio*observed(table, order)[denominator] for order in ORDERS)


def enumerate_moments(population, ratio, numerator='C', denominator='A'):
    """Exhaust every assignment for a SMALL known potential population.

    This is an independent analytic fixture, never a way to invent unobserved
    potential outcomes for actual VM measurements.
    """
    tables = checked_population(population)
    n = len(tables)
    if n > 5:
        raise ValueError('exact enumeration is limited to five cohorts')
    contrasts = [potential_contrasts(t, ratio, numerator, denominator) for t in tables]
    deltas = [sum(c)/6 for c in contrasts]
    target = sum(deltas)/n
    vectors = list(product(*contrasts))
    means = [sum(v)/n for v in vectors]
    variance = sum((m-target)**2 for m in means)/len(means)
    estimated = sum(sum((v-m)**2 for v in vector)/(n*(n-1)) for vector, m in zip(vectors, means, strict=True))/len(means)
    heterogeneity = sum((d-target)**2 for d in deltas)/(n*(n-1))
    independent_variance = sum(sum((v-d)**2 for v in c)/6 for c, d in zip(contrasts, deltas, strict=True))/n**2
    return dict(assignments=len(vectors), expectation=sum(means)/len(means), target=target,
                actual_variance=variance, expected_estimated_variance=estimated,
                independent_variance=independent_variance, heterogeneity=heterogeneity)


def assignment_schedule(draws, phase):
    """Consume recorded independent draws without searching or enforcing balance."""
    if phase not in ('aa', 'compare'):
        raise ValueError('unknown phase')
    if not isinstance(draws, list) or len(draws) != COHORTS*10 or any(type(v) is not int or not 0 <= v < 6 for v in draws):
        raise ValueError('exactly600 integer draws in[0,6) required')
    rows = []
    groups = [(i, policy) for i in range(5) for policy in POLICIES]
    for cohort in range(COHORTS):
        offset = cohort % 10
        visit = groups[offset:] + groups[:offset]
        if (cohort // 10) % 2:
            visit = visit[::-1]
        for index, policy in visit:
            draw_index = cohort*10 + index*2 + POLICIES.index(policy)
            for position, role in enumerate(ORDERS[draws[draw_index]]):
                rows.append(dict(name=f'c{cohort:02d}-{CASES[index]}-{policy}-{role}', cohort=cohort,
                                 case=CASES[index], case_index=index, policy=policy, role=role,
                                 position=position, draw_index=draw_index))
    return rows
