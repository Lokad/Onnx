# Prospective e5 comparison across fresh processes

This is preparation for a distinct experiment, `e5-fresh-process-uncertainty-v1`.
**No new e5 VM comparison has run under this protocol.** The old failed
[sequential](../fingerprint-deployment/aa-results-20260920.md) and
[resident](../interleaved-processes/aa-results-20260920.md) experiments keep their
original verdicts. The [resident-data diagnostic](../resident-variation/results-20260921.md)
shows why collecting additional calls in the same process is insufficient.

The [local preparation report](preparation-results-20260921.md) records60passing
functional workers, ten mathematical tests and the synthetic dependence failure.
Statistical readiness and the complete campaign runner remain unfinished.

## What changes

Each case and public policy has sixty cohorts. Each cohort contains three fresh,
sequential processes: current managed defaults (A), candidate (C), and ORT (N).
For A/A, C also uses current defaults. Only a passing A/A permits the separately
frozen candidate phase, where C enables the already qualified fingerprint-string
cache and final wide LayerNorm transform. Product defaults remain unchanged.

Both phases cover lengths8,30,padded128,128,512 and Default/Memory. Each phase has
**1,800 processes**. All six role orders occur once in each six-cohort block for
each case/policy. A single published hash seed fixes the schedule; seeds are not
searched. The phases have separate orderings. Processes inherit CPU2 before CLR
startup and use normal .NET10.0.8 settings on the AMD VM. The native baseline is
the existing pinned e5 ORT1.23.2 library, not the audio suite's ORT1.29.0.

Conditioning retains at least128 calls and30 cumulative Execute seconds. Each
process then retains sixteen complete measured blocks of32/16/4/4/2calls,
respectively. That is **334,080 measured calls per phase**, plus every first and
conditioning call. Conditioning alone costs at least15hours per phase; actual
elapsed time includes import, measurement and validation. Sixty cohorts is a
fixed resource budget, not a guarantee of adequate precision. A failed or
inconclusive phase does not trigger an automatic larger-count rerun.

## Estimator and limitations

`estimator.py` reduces all measured integer ticks in each process to one exact
rational arithmetic mean. The uncertainty sample size is sixty paired process
means, regardless of the number of calls. The reported ratio is the ratio of
arithmetic means, not a geometric mean or a mean of per-cohort ratios. No call,
tail, process or GC event is dropped.

For denominator mean x and numerator mean y, invert:

    (y-r*x)^2 <= t^2 * (vy - 2*r*cov + r^2*vx)

Here vx,vy and cov are the variances/covariance of the two means, computed across
cohorts. This paired Fieller confidence set includes uncertainty in the
denominator. A nonpositive quadratic leading coefficient is explicitly unbounded
and cannot support a bounded claim. Exact rational moments preserve cancellation
for proportional observations. Student t critical values are computed using a
trigonometric integral and tested against closed forms and independent density
quadrature.

The nominal family coverage is95%, using Bonferroni over20 A/A contrasts or60
comparison contrasts: five cases × two policies × two boundaries, with C/A in
A/A and C/A,C/N,A/N in comparison. **This is conditional on independent cohorts
and a sufficiently normal paired contrast.** Sixty samples, balanced order and
passing A/A do not prove those assumptions. Persistent temporal dependence or
unresolved systematic movement must withhold a confidence interpretation; timings
must not be normalized to make it pass. The synthetic sensitivity tool deliberately
includes a serially correlated scenario to illustrate that limitation.

The higher-level repetition principle and independent-system ratio intervals are
discussed by [Kalibera and Jones2013](https://kar.kent.ac.uk/33611/45/p63-kaliber.pdf).
The paired covariance formula above follows directly from the displayed contrast
inequality; it is not presented as their independent-system formula.

## Prospective screens

Every A/A C/A interval must contain1 and be contained in[.99,1.01]. In comparison,
every C/A upper bound must be at most1.01; the8-token upper bound must additionally
be at most.98 and the30-token bound at most.99. Primary parity is a separate claim:
every primary C/N upper bound for the stated public policy must be at most1.05.
Length512 remains regression/stretch coverage. Both public Execute/ORT Run and
the enclosing request including Reset are reported.

These statistical screens do not replace full native output validation at1e-4,
input/held-output ownership, actual binary/flags/runtime identity, process births,
resource limits, foreign CPU accounting or assumption review. The evaluator
always reports `ready_for_promotion=false`: a statistical calculation alone cannot
approve a product change. No campaign runner or frozen deployment exists yet.

## Local checks

From the repository root:

    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/e5/process-uncertainty -p test_*.py -v
    C:/Python313/python.exe -X utf8 -B tests/e5/process-uncertainty/simulate.py --output artifacts/e5-process-uncertainty-20260921/sensitivity.json

The unit tests cover exact process means, retained tails, Fieller inversion,
scale/reciprocal properties, paired covariance, unbounded denominators, independent
t quantiles, complete locally balanced schedules and damaged/missing/duplicate
process refusals. The sensitivity command refuses an existing output and makes
no VM inference or performance claim.

`Program.cs` is an isolated public-API probe referencing the current qualified
core DLL. `smoke.py` runs sixty reduced local functional workers across both
phases, all five cases, both policies and all three roles. Those smokes are not
A/A timing qualification. Full campaign preparation, frozen artifact identity,
resource audit and independent report verification remain required before launch.
