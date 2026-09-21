# Randomized e5 runner: local integration complete

All **12 local functional workers** pass full-output, ownership, input, flag,
native-library and resource checks:48measured,80conditioning and12first calls.
Ten damaged manifests refuse. An additional worker given a mismatched model
identity stops before import or inference; its runner records a terminal failure
and cleans up owned processes. All30recorded process identities are terminal.

The runner reuses the unchanged, qualified `ProcessUncertainty.dll` and its v1
measurement specification within the enclosing v2 campaign. The earlier
[sixty-worker qualification](../process-uncertainty/preparation-results-20260921.md)
remains the complete five-case/two-policy producer evidence. This new Windows
integration smoke does not exercise the Linux deployment or provide timing claims.

## Actual assignments

Both schedules were drawn once after design commit3558c8c, before new e5 inference.
The1,200independent `secrets.randbelow(6)` draws specify1,800workers per phase.
Realized role-position counts range11–29in A/A and14–28in comparison; there was
no balance-based redraw. The record freezes assignments only.

| Receipt | Bytes | SHA256 |
|---|---:|---|
| assignments-frozen.json | 889,069 | `2ac2bd42acfc811b39f2df099c7e93f31789bea18236e0e507d74e60ba5b16f5` |
| local-payload/frozen.json | 52,180 | `a6ad1364a457483f7b3b4462c330faf6aec592090bcde667d6998d83cf5463cc` |
| runner-verification.json | 30,208 | `548abbea45234903e21d7ca69e20c079f6d9012cc6fc92ebd8c31922f5fcc5cb` |

The artifact is `artifacts/e5-randomized-processes-20260921`. All168files bound by
the runner receipt were rechecked.

## Corrections retained

The nine-test suite caught a rounding error in the largest-variance-share
diagnostic: exactly proportional means reported a nonzero share when subtracting
a floating-point fitted ratio. The diagnostic now uses its exact rational ratio;
Fieller estimates and intervals were unaffected. All nine tests pass.

The runner also now marks failure terminal. The negative worker exercises this
path. Source comparison verifies these are the only two changes after the
positive smoke: the analysis diagnostic and the failure-state marker. The
successful execution path is unchanged.

After validating the negative worker, the local verifier failed on an encoding
name typo. The original source and error are retained. Correcting `utf8-sig` to
`utf-8-sig` allowed verification to resume from terminal evidence. No worker was
restarted. All completed artifact writers should remain closed.

## Next execution step

Prepare the full payload/runtime bindings and independent collection/report
verifier. The AMD Whisper comparison remains the VM's sole inference workload
until its actual processes are terminal. Full e5 raw arrays exceed remaining VM
disk space; use verified `/dev/shm` capacity and collect the completed phase before
the conditional comparison. The runner requires3GiBfree at phase start, retains
a512MiBevidence-space guard, and has a30hour phase ceiling.

The [method's assumptions](README.md) remain explicit. No new VM e5 measurement
or default promotion follows from this local integration result.
