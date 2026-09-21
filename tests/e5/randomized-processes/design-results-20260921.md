# Independent-assignment e5 design: exact proof and sensitivity

Six mathematical/schedule tests pass. Exact enumeration verifies unbiased
contrasts and the conservative expected variance identity for independent
three-role assignments, including heterogeneous potential outcomes. All
**14,000 synthetic trials** retain their potential tables and sixty assignment
draws. A separate60-digit Decimal implementation recomputes every ratio and
interval: **42,000 endpoint/ratio values** agree within2.23e-16.

These results support the implementation of the
[prospective method](README.md), subject to its stated assumptions. They do not
qualify a product default, establish exact confidence coverage, or measure e5.
The prior dependent-outcome simulation remains unchanged.

## Fixed-population results

Each scenario contains sixty cohorts with three positions and three possible
engine latencies at each position. All tables are constructed before assignments.
There are2,000 independently randomized trials per scenario. Intervals use the
A/A marginal nominal level99.75%; no whole-family empirical coverage is claimed.
The target is computed from all potential outcomes in the finite scenario.

| Scenario | Finite target C/A | Intervals containing target | Median half-width |
|---|---:|---:|---:|
| Common serial variation | 1.000000 | 1,989 / 2,000 | 0.167% |
| Common trend and step | 1.000000 | 1,995 / 2,000 | 0.199% |
| Persistent position costs | 1.000000 | 1,994 / 2,000 | 1.473% |
| Separate serial role trajectories | 1.017795 | 2,000 / 2,000 | 0.477% |
| Heterogeneous candidate effects | 0.970312 | 2,000 / 2,000 | 0.332% |
| One dominant, partly unobserved effect | 1.011104 | 725 / 2,000 | 0.176% |

The first five scenarios illustrate behavior with fixed drift or heterogeneous
effects. The role-serial scenario's finite target is1.017795, even though its
generating role deviations have zero long-run expectation. Comparing its
intervals with1 would answer a different question. Approximate coverage is not
exact, including the1,989/2,000 result in the common-serial scenario.

The persistent position costs produce intervals too wide for the declared A/A
equivalence screen: none of its2,000single contrasts passes. Sixty cohorts do not
guarantee adequate precision, and the design must not increase counts until a
screen passes.

The dominant-effect counterexample makes one candidate potential latency three
times its usual value. Most assignments never observe that potential. Only725
intervals contain the true finite ratio;1,273nevertheless pass the A/A screen.
This exposes the large-sample regularity assumption. A/A and narrow intervals
alone cannot rule out unobserved exceptional effects.

The seventh scenario deliberately introduces carryover: the worker immediately
after C is15%slower. It violates the fixed-potential-outcome model. Only3/2,000
intervals contain the no-carryover reference.97; the observed mean ratio is.923760.
That count is an interference diagnostic, **not confidence coverage under the
declared model**, because the reference no longer describes its observation
mechanism. These failures are retained with the successful arithmetic checks.

## Evidence and next step

Artifact: `artifacts/e5-randomized-processes-20260921`.

| Receipt | Bytes | SHA256 |
|---|---:|---|
| results.json | 4,002 | `1f3aab8906734b84293792ef601508106856a6830be33c6eb07833c8f409debc` |
| simulation-verification.json | 821 | `fff9baad7d18e2112c0c22dff49d0096a7476b0a125c70b07ad381082d146bda` |

The simulation verifier uses sums of squares and60-digit quadratic roots,
independently of the producer's rational centered moments. It checks every
retained draw against the scenario's fixed simulation generator, every finite
target and all report aggregates. Simulation random seeds are fixtures; actual
campaign assignments require their own one-shot recorded random draws before
inference. A complete campaign runner, frozen runtime/binary bindings and raw
evidence audit remain necessary. The matched Whisper job remains live on the VM.
