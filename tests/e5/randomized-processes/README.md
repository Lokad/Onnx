# Independently randomized fresh e5 processes

`e5-independent-assignment-v2` replaces the proposed locally balanced assignment
scheme before any new e5 VM inference. It retains sixty cohorts per case/policy,
three fresh sequential processes per cohort, both public policies and all five
cases. **No VM campaign has started under this design.**

The [previous prototype](../process-uncertainty/preparation-results-20260921.md)
retains its completed local checks and failed dependence sensitivity. Its exact
Fieller implementation and qualified producer remain reusable. This design
changes the assignment mechanism and the interpretation of uncertainty.

## What the randomization establishes

Each cohort has three execution positions. Independently assign A (current
managed defaults), C (identical defaults in A/A; both optional changes enabled
in comparison), and N (native ORT) to those positions, choosing uniformly among
the six permutations. Record every choice before inference. Exact role-position
balance is not imposed, and unfavorable position counts do not cause a redraw.
Case/policy order rotates through ten groups, reversing every ten cohorts.

For inference, each position has a fixed potential latency under each engine.
Only the assigned engine's latency is observed. These potential latencies may
have common drift, different means and role-specific temporal patterns. The
independent assignments supply the randomness; the latency trajectories need not
be independent draws from a population.

For a trial ratio r, let d_i = C_i - r*A_i and delta_i = E(d_i) over the six
assignments in cohort i. Independent assignments imply:

    Var(mean(d)) = sum(Var(d_i)) / n²
    E(sample_variance(d) / n)
      = Var(mean(d)) + sum((delta_i - mean(delta))²) / (n*(n-1))

The extra term is nonnegative. `test_design.py` proves these identities with
exact rational arithmetic by exhaustively enumerating 216 assignments for
heterogeneous three-cohort examples and all three engine contrasts. The
estimated variance is conservative **in expectation**. Large-sample coverage
still requires regularity; this identity alone is not a finite-sample confidence
guarantee.

The unchanged Fieller calculation inverts the approximate test for mean(d)=0.
The ratio being estimated is the ratio of arithmetic mean potential latencies
over this campaign's execution positions. It is **not a guarantee about a
long-run population or a future deployment**. Student59 critical values are used
as an approximation; the contrasts do not have an exact Student distribution.

[Fogarty2017, section3](https://arxiv.org/pdf/1612.05179) describes fixed-population
paired inference and its conservative variance estimator. The three-role
extension above is our direct derivation and enumeration, without regression
adjustment or a claim to implement that paper's specific theorem.

## Assumptions that remain

Potential outcomes must be unaffected by earlier assignments. Independent
processes, bounded resources and thirty seconds of conditioning reduce shared
state, but cannot prove absence of thermal, allocator, operating-system or other
carryover. A single exceptional treatment effect must not dominate the campaign.
The synthetic tests include explicit violations of both assumptions; their
intervals fail badly. Observed diagnostics cannot reveal all unobserved potential
outcomes, so these limitations must accompany any confidence statement.

Before deployment, the observed-variance guard is fixed at 20%: no one cohort
may contribute more than one fifth of the sum of squared fitted contrast
residuals. A failure withholds the confidence interpretation and stops the
conditional comparison. Passing this operational guard cannot establish the
regularity of unobserved potential outcomes. Assignment draws, worker counts
and performance thresholds are unchanged.

The approximate family confidence remains95%, using Bonferroni for twenty A/A
contrasts and sixty comparison contrasts. Every A/A C/A interval must contain1
and lie within[.99,1.01]. Candidate C/A upper bounds remain.98at8tokens,.99at30,
and1.01otherwise, at both timing boundaries and both policies. Primary parity
requires each primary C/N upper bound at most1.05 for the stated policy;512is
regression/stretch coverage. No statistical screen alone changes defaults.

## Local validation and pending deployment

The [sensitivity report](design-results-20260921.md) retains fourteen thousand
trials and all draws, plus independent Decimal recomputation. No actual VM
timings were used to select the method, assignment draws, counts or thresholds.
The [runner report](runner-results-20260921.md) records twelve passing local
workers, ten manifest refusals, the terminal failure-path test and nine passing
mathematical/contract tests.

Run the read-only tests from the repository root:

    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/e5/randomized-processes -p test_*.py -v

The completed simulation and verification outputs are under
`artifacts/e5-randomized-processes-20260921`. Their writers refuse existing paths;
do not rerun them into that artifact. `freeze_assignments.py` has recorded both
actual schedules once, after commit3558c8c, with no seed or balance search. The
manifest SHA256 is2ac2bd42acfc811b39f2df099c7e93f31789bea18236e0e507d74e60ba5b16f5.
The runner and raw-output/resource auditor are implemented and locally exercised.
The deployment, collection and full-report tools are implemented. Eleven tests
pass, including full-count synthetic tick reports for both phases and independent
80-digit Decimal interval calculations. No model inference occurs in those tests.

The worker measurement contract can reuse the already verified
`ProcessUncertainty.dll` unchanged. Its raw specification retains the v1 worker
identifier; the enclosing v2 campaign explicitly supplies the new assignment
manifest. This avoids rebuilding or repeating unchanged producer qualification.
Each full phase still has1,800workers,334,080measured calls and at least15hours
of conditioning. The ongoing Whisper comparison remains the sole VM workload.

## Deployment and reporting

Commit tools before running `prepare.py`. It copies the qualified worker and
inputs unchanged, verifies the existing qualification receipts, and writes the
local payload once. `stage.py` refuses to launch until the existing Whisper
completion monitor and its final verification have succeeded and all preceding
process identities are terminal. It binds actual runtime files and deploys into
the new `/dev/shm/lokad-e5-independent-20260921` directory; ordinary VM disk space
cannot accommodate the complete evidence.

From the repository root, use these commands sequentially:

    C:/Python313/python.exe -X utf8 -B tests/e5/randomized-processes/prepare.py
    C:/Python313/python.exe -X utf8 -B tests/e5/randomized-processes/stage.py
    C:/Python313/python.exe -X utf8 -B tests/e5/randomized-processes/observe.py --phase aa

Observe the existing job until its recorded PID and birth time are terminal.
Then collect and verify all raw arrays and calls:

    C:/Python313/python.exe -X utf8 -B tests/e5/randomized-processes/collect.py --phase aa
    C:/Python313/python.exe -X utf8 -B tests/e5/randomized-processes/report.py --phase aa
    C:/Python313/python.exe -X utf8 -B tests/e5/randomized-processes/verify_report.py --phase aa

Collection streams the archive to local disk. It does not duplicate the complete
raw evidence on the VM. The report supplies all means, medians, tails, process
and position diagnostics, allocations and GC counts. The independent verifier
recomputes intervals from integer ticks and checks every displayed table row.
A failed raw audit is retained without a scored report. A complete report with
failed statistical or diagnostic screens remains publishable as a failure.

Only an independently verified passing A/A produces `aa-gate.json`.
`start_compare.py` checks that gate and all retained identities before launching
the already frozen comparison. Observe, collect, report and verify it using
the same commands with `--phase compare`. No script changes product defaults.
Writers refuse existing paths. A timeout is not termination: inspect the actual
recorded process and preserve completed stages before recovering a failed transfer
or report. Never restart completed inference to recover reporting.
