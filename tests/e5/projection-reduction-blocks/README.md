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
