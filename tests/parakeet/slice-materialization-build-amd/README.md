# Build and qualify the guarded slice reshape copy

This is the corrected v2 namespace. The first run passed all 25 copy cases and
the identity check, but seven source-policy tests could not discover the source
root from a runtime directory outside that tree. Preserve its failed closure
`84358a71`. V2 places the test runtime under the source root and replaces the
helper's forbidden null-forgiving operator with an annotated nullable out result.
The candidate mechanism, fallback and test expectations stay the same.

Qualification is now complete: [369/369 cases pass in both verified modes](../slice-materialization-results/qualification-20260924.md).
The v2 disabled-mode attempt used an ineffective `DOTNET_EnableAVX512F` name and
its identity test correctly failed. `correct_mode.py` reused the same binaries
under a new namespace with `DOTNET_EnableAVX512=0`; all tests then passed.
Both failed stages remain closed. Do not rerun the obsolete build/capture flow.
The corrected-mode stage is also terminal; its commands were `prepare`, `stage`,
`launch`, `observe`, `collect`, `audit`, with the same Python invocation below.

This stage consumes the immutable 423-file candidate snapshot after all 1,920
actual layouts passed. Only `TensorSlice.Reshape` changes, with one private helper;
25 copying contract cases are added. The build adds one campaign-only identity
test, for 369 tensor tests including the existing 343. It verifies the consumed
Core hash, one visible processor and actual AVX-512 support in each mode.

Build first, collect and review exact compiled scope; only an admitted review
permits the separate test stage. Preserve every original method and flag except
Reshape, add only the private helper, and keep public surface equal. The complete
tensor suite runs with AVX-512 enabled, then disabled. Any failure stops the stage.
This is correctness qualification, with no admitted performance result.

Use the same `run.py prepare`, `stage`, `launch build`, `observe build`, `collect
build`, `review_build.py`, `launch capture`, `observe capture`, `collect capture`,
`audit.py` flow as the layout stage, under `C:/Python313/python.exe -X utf8 -B`.
Namespaces are exclusive-create and closed stages must not be observed again.

CPU2 runs every child; CPU0 monitors exact PID/birth ownership. Start each command
with 2 GiB available and 1 GiB tmpfs; retain at least 1 GiB of both. Owned RSS is
capped at 3 GiB, total stage output at 512 MiB, builds at 180 seconds per command
and each test process at 300 seconds. Reuse the pinned SDK, offline feed and
selected runtime. No Windows build/inference or extra model copy is required.

Artifacts: `artifacts/parakeet-slice-materialization-build-amd-v2-20260924` and
`/dev/shm/lokad-parakeet-slice-materialization-build-v2-20260924`.
