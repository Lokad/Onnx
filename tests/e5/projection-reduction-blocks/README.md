# Reduction blocking with existing packed weights

This standalone correctness prototype divides the existing twelve/eight-row
packed kernel's reduction into fixed128/256-position blocks. Each block visits
all corresponding row groups before moving on. Input row stride remains the
full reduction length; weights keep the existing32-column layout. There is no
activation copy, new packing format or allocation in the candidate. Original
two/three-row tails, nonzero destination accumulation and FMA order remain.

Pinned ORT1.23.2's packed SGEMM uses256-position reduction blocks. Voice-branch
commits cbcfc65/529e78f/a09b0a3 explore this idea with other packing/gather changes;
this prototype does not import those product changes. It is distinct from the
closed activation-packing experiments. No cache or speedup claim follows from
source structure alone.

From the repository root:

    C:/Python313/python.exe -X utf8 -B tests/e5/projection-reduction-blocks/test_generate.py
    C:/Python313/python.exe -X utf8 -B tests/e5/projection-reduction-blocks/generate.py --output artifacts/reduction-proof-new/source
    dotnet build artifacts/reduction-proof-new/source/Probe.csproj -c Release --tl:off --nologo -v minimal -p:FrozenCorePath=<absolute-qualified-core> -o artifacts/reduction-proof-new/bin
    dotnet artifacts/reduction-proof-new/bin/Probe.dll --host

The local AVX2 host can check source transformations, compilation and refusal.
Actual arithmetic proof requires the AMD AVX-512 host, with CPU2 inherited before
runtime startup. Give the executable one new output directory. All473 geometries
retain full outputs; both block sizes must match every original bit, including
guard regions and a second accumulation. Independent scalar-FMA coordinates,
exceptional values, input/weight preservation and31 refusal cases are checked.

Freeze sources/binaries before AMD proof. Do not add VM load while an original
timing campaign runs. Supervise proof on CPU0 with fixed180seconds/2GiB sampled
process-group RSS/1GiB available memory, retaining every sample and process birth.
Timing is not implemented; a separate fixed protocol must precede candidate
timing. See `.agent/m2-reduction-blocks-20260920.md`.

The AMD proof reuses the closed v3 local binary rather than rebuilding or rerunning
local checks. After committing tools, prefix these commands with
`C:/Python313/python.exe -X utf8 -B`:

    tests/e5/projection-reduction-blocks/prepare_proof.py --artifact artifacts/e5-reduction-blocks-proof-20260920
    tests/e5/projection-reduction-blocks/vm.py launch --artifact artifacts/e5-reduction-blocks-proof-20260920
    tests/e5/projection-reduction-blocks/vm.py poll --artifact artifacts/e5-reduction-blocks-proof-20260920
    tests/e5/projection-reduction-blocks/vm.py collect --artifact artifacts/e5-reduction-blocks-proof-20260920
    tests/e5/projection-reduction-blocks/audit_proof.py --artifact artifacts/e5-reduction-blocks-proof-20260920

The plain and instrumented runs each execute all473 cases. Every candidate output
and second accumulation is checked against the original inside the pinned probe;
complete original first outputs and hashes of second outputs are retained.
The independent auditor checks all saved bytes/guards and exact equality between
the two processes. It does not pretend that candidate arrays were separately
saved. Both twelve/eight-row bodies must be FullOpts, with24/16 FMAs and12/8
broadcasts in the reduction loop, no loop calls or vector stack accesses.
Scalar stack addressing remains visible in the complete dump. Original row
stride versus separate reduction count also requires direct instruction review.

Five local test methods cover source drift, block-boundary coverage, eleven
damaged telemetry records and six damaged/missing code captures. The proof
runner reuses the existing Linux process-group guard. Collection preserves all
records and checks terminal PID/start ticks before writing its manifest.
