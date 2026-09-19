"""Frozen empirical product comparison, without confidence or ORT claims."""
import itertools
import math
import statistics

CASES = ('e5-8tok', 'e5-30tok', 'e5-30pad128', 'e5-128tok', 'e5-512tok')
ROLES = ('controlA', 'controlB', 'candidate')
CRITERIA = dict(control_aggregate=.01, control_each=.03, padded_gain=.01,
                padded_each_max=1.02, other_aggregate_max=1.02,
                other_each_max=1.05, foreign_max=.02, steal_max=.005)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def schedule():
    result = []
    permutations = list(itertools.permutations(ROLES))
    for visit in range(6):
        order = list(CASES[visit % 5:] + CASES[:visit % 5])
        if visit % 2:
            order.reverse()
        for name in order:
            triplet = len(result) // 3
            for position, role in enumerate(permutations[(visit + CASES.index(name)) % 6]):
                result.append(dict(name=name, role=role, visit=visit,
                                   triplet=triplet, position=position))
    return result


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def evaluate(workers, telemetry):
    require(len(workers) == 90, 'Expected ninety workers')
    for worker, planned in zip(workers, schedule(), strict=True):
        require(all(worker[k] == v for k, v in planned.items()), 'Schedule differs')
        for boundary in ('execute', 'request'):
            samples = worker[boundary]['samples_ms']
            require(len(samples) == 33 and all(positive(t) for t in samples),
                    'Invalid measured samples')
    for name in ('maximum_foreign_cpu_fraction', 'maximum_steal_fraction'):
        value = telemetry[name]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                'Invalid machine accounting')
    health = dict(foreign_cpu_at_most_2_percent=telemetry['maximum_foreign_cpu_fraction'] <= .02,
                  steal_at_most_half_percent=telemetry['maximum_steal_fraction'] <= .005)
    cases = {}
    control_passed = candidate_passed = True
    for name in CASES:
        cases[name] = {}
        for boundary in ('execute', 'request'):
            means = {role: [statistics.mean(w[boundary]['samples_ms']) for w in workers
                            if w['name'] == name and w['role'] == role] for role in ROLES}
            aggregate = {role: statistics.mean(v) for role, v in means.items()}
            controls = [b / a for a, b in zip(means['controlA'], means['controlB'], strict=True)]
            candidates = [c / ((a + b) / 2) for a, b, c in
                          zip(means['controlA'], means['controlB'], means['candidate'], strict=True)]
            control_ratio = aggregate['controlB'] / aggregate['controlA']
            candidate_ratio = aggregate['candidate'] / ((aggregate['controlA'] + aggregate['controlB']) / 2)
            control = dict(aggregate_within_1_percent=abs(control_ratio - 1) <= .01,
                           every_triplet_within_3_percent=all(abs(r - 1) <= .03 for r in controls))
            if name == 'e5-30pad128':
                candidate = dict(aggregate_gain_at_least_1_percent=candidate_ratio <= .99,
                                 every_triplet_ratio_at_most_1_02=all(r <= 1.02 for r in candidates))
            else:
                candidate = dict(aggregate_regression_at_most_2_percent=candidate_ratio <= 1.02,
                                 every_triplet_regression_at_most_5_percent=all(r <= 1.05 for r in candidates))
            control_passed &= all(control.values())
            candidate_passed &= all(candidate.values())
            cases[name][boundary] = dict(mean_ms=aggregate, visit_mean_ms=means,
                control_ratio=control_ratio, candidate_ratio=candidate_ratio,
                control_visit_ratios=controls, candidate_visit_ratios=candidates,
                control_criteria=control, candidate_criteria=candidate)
    return dict(passed=bool(control_passed and candidate_passed and all(health.values())),
                control_passed=bool(control_passed), candidate_passed=bool(candidate_passed),
                health=health, cases=cases)

