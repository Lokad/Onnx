# Complete-call screen of actual Winograd products

This protocol is fixed before timing. Buildcc8d46c5 confines the product change
to MultiplyWinograd512. Numerical closure47513c48 proves exact old and expanded
rows for both actual DLLs. Codegen reviewbf998641 admits paired broadcast reuse,
including its437byte method growth and loss of caller inlining. Capture116194a1
remains failed for interleaved output; method-specific correction74a30abd supplies
all missing tiers. No product change or speed result has been admitted yet.

The exact current Core521bae17 and candidate Core90b2164b are bound by typed
delegates. Both roles execute PrepareWinograd, PlanWinograd and ExecuteWinograd.
No arithmetic implementation is compiled into this consumer. A separate
AssemblyLoadContext per loaded role keeps the same-name DLL identities distinct;
the verification process loads both and every scored process loads only its role.
Actual loaded paths and hashes are recorded and checked outside the call timer.

Both roles time output allocation, scratch planning/pool rentals/returns,
validation, finite checks, runtime transforms, multiplication, layout conversion
and identical bias/residual/ReLU epilogues. The returned array is independently
owned. Input loading, weight preparation, hashing, assertions and serialization
are outside the call timer. All preparation clocks are retained separately.
This is the complete convolution call, excluding graph/audio scheduling and
supplying no new application or Microsoft ORT timing.

Every87 eligible captured stride-one call and all eight forms remain in their
original order and multiplicity. The fixed geometry rule ceil(2^31/direct work)
gives three iterations per call. Each process executes one warmup and three
measured passes:1,044calls,261warmup and783measured. Four fresh processes run
current-a,candidate-a,candidate-b,current-b, retaining4,176clocks/3,132measured
and464preparation clocks for all29weight sets before each pass. No calibration,
trimming, omissions or adaptive repetition. Both products must match the same
qualified Winograd output hashes, preserving all input/weight/held-output checks.

The original gates.py is byte-identical. Exact clock fractions require at least
10% aggregate improvement and at most5% regression in each form. All18controls
must pass: max/min process means<=1.10 aggregate and<=1.20 per form for each role.
Both candidate aggregate process means must be strictly below both current means.
An unchanged failed timing trial is never repeated to seek a favorable result.

Eight serial jobs: SDK, restore, Release build, a separate174call verification
through both actual DLLs, then the four scored processes. Verification must pass
before timing. Source-scope generation preserves every timing loop and clock
boundary, replacing only the direct/Winograd pair with two actual Winograd DLLs
and the necessary identity/scratch/hash checks. Score tests reject wrong products,
old direct-convolution costs/outputs, missing clocks and every original gate fault.

SDK10.0.204/runtime10.0.8, CPU2 workers/CPU0 monitor, ordinary AVX512, no profiler
or ISA/tiering/inlining overrides. Build preflight10GiB available; verification
and timing12GiB. Tmpfs preflight3GiB; live8GiB RSS,1GiB available/tmpfs minimum,
900seconds/job,1GiB output and2GiB total artifacts. Retain all failures.

From repository root use C:/Python313/python.exe -X utf8 -B with test_score.py,
run.py prepare/stage/launch/observe/collect and audit.py. Collect only terminal
identities. Refuse existing artifacts/pyannote-winograd-output-blocks-screen-amd-20260923
and /dev/shm/lokad-pyannote-winograd-output-blocks-screen-20260923. Publish every
call/preparation clock and every gate. Only admission permits separate complete
product/application qualification under the existing3% dialogue improvement,
5%crop regression and12control gates; no root change before full admission.
