# Fresh-process e5 estimator and producer: local preparation

The proposed [fresh-process comparison](README.md) now has a working estimator,
balanced schedule and isolated public-API producer. **All 60 local functional
workers pass**, covering both phases, five cases, two public policies and three
engine roles. No new VM e5 comparison has run, and neither optional optimization
has been enabled by default.

The producer builds with SDK10.0.204, zero warnings and zero errors. It references
the already qualified current core DLL rather than rebuilding the product.
Against the earlier e5-qualified source `4f10e8b`, the only current core change is
the eleven-line additive explicit-budget `CreateExecution` overload. The public
Execute path used by this probe is unchanged by that addition. Actual output,
input, held-output, flags, cache and native-library checks run in each worker.

## Functional and mathematical evidence

The Windows smoke uses .NET10.0.12, inherited CPU2 and the existing ORT1.23.2
library. It retains 240 measured, 298 conditioning and 60 first calls. These small
functional runs provide no performance comparison. All 61 supervisor/worker birth
identities are terminal. A separate scalar-array implementation recomputes
**7,428,096 output comparisons**, and all managed outputs match across default
and candidate roles for each case. All 24 deliberately damaged records refuse.

Ten mathematical/schedule tests pass. They check Fieller roots by independently
evaluating the paired t inequality, Student critical values against closed forms
and density quadrature, scale and reciprocal identities, covariance, unbounded
denominators, retained tails, one mean per process and missing/duplicate/rounded
process rejection. Each prospective phase contains exactly 1,800 fresh workers
and 334,080 measured calls. Every case/policy gets sixty workers per role, with
each role twenty times in each position and all six orders in every block of six
cohorts. The A/A and comparison critical values are3.158537494661 and3.522069119499,
respectively, for59degrees of freedom and the declared confidence families.

## Synthetic sensitivity exposes a readiness limit

The fixed simulation uses 1,000 trials per scenario, sixty pairs per trial, a true
ratio of1 and3% independent common cohort variation. Each row evaluates one C/A
contrast at the A/A marginal confidence level99.75%; it does not simulate the
whole twenty-contrast family's pass probability.

| Process variation scenario | True ratio covered | Single contrast passes A/A | Median interval half-width |
|---|---:|---:|---:|
| Independent, 0.5% standard deviation | 997 / 1,000 | 997 / 1,000 | 0.286% |
| Independent, 1% standard deviation | 998 / 1,000 | 975 / 1,000 | 0.573% |
| Independent, 2% standard deviation | 995 / 1,000 | 4 / 1,000 | 1.145% |
| Serial correlation0.85, 1% standard deviation | 605 / 1,000 | 510 / 1,000 | 0.502% |

The independent scenarios exercise expected behavior under their synthetic
assumptions. The serial scenario deliberately violates cohort independence and
shows severe undercoverage despite narrow reported intervals. Therefore passing
mathematical tests or an A/A screen cannot establish confidence validity on the
VM. Also, sixty cohorts will often be insufficient if independent process
variation is around2%. These are sensitivity findings, not estimates of actual
VM variance, coverage or required repetition.

Before any deployment, the design still needs an explicit defensible treatment
of dependence, a complete campaign runner/resource auditor and an independent
report verifier. Current preparation does not authorize the confidence
interpretation or a default change. The fixed budget has at least15hours of
conditioning per phase; it must not become an automatic sequence of larger
retries. The ongoing Whisper job remains the VM's sole inference workload.

## Retained evidence

The local artifact is `artifacts/e5-process-uncertainty-20260921`.
`local-verification.json` binds388 retained files, all independently rechecked.
The build, sixty raw worker outputs, process samples, full before/after arrays,
source bridge and sensitivity results are retained. No existing evidence writer
needs to run again.

| Receipt | Bytes | SHA256 |
|---|---:|---|
| smoke-audit.json | 65,216 | `8aa3e3e09200ac63ca496cb3e79d42f924b2ad1e3af015ab4220662e7e3bd7a0` |
| local-verification.json | 69,851 | `2621d52b55a1f6a2d0b259f61feb3e7b9d48137c189d30447d8ea655d5cc6620` |
| sensitivity.json | 2,374 | `32ffdde4110e7a6196b76cec642302237964377b1c7e366edadc656ff525497c` |

The exact qualified core SHA256 is
`d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4`.
The full prospective definitions and their statistical assumptions are in the
[README](README.md); the focused execution plan is
`.agent/m1-process-uncertainty-20260921.md`.
