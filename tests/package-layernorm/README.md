# Current LayerNorm package consumption

This verifies the distinct packaging boundary after the LayerNorm product change.
It reuses the complete Windows/AMD product qualification through exact source
equivalence, and runs an independent app restored from a new private feed/cache.
Nothing is published and no old full-model campaign is repeated.

From the repository root, prefix each command with
`C:/Python313/python.exe -X utf8 -B`:

    tests/package-layernorm/prepare.py --artifact artifacts/layernorm-package-20260920
    tests/package-layernorm/run.py --artifact artifacts/layernorm-package-20260920
    tests/package-layernorm/audit.py --artifact artifacts/layernorm-package-20260920

Commit sources before preparation. The source archive is checked against every
Git blob, permitting only exact text CRLF conversion. All production/test/build
inputs must match qualified `4f10e8b`. The runner calls the original `pack.cmd`,
then restores/builds the retained app from its private feed and runs four fresh
processes: default, fingerprint-only, wide-LayerNorm-only and both flags.

The consumer preserves Relu and MNIST file/bytes/sliced-memory import checks,
adds eight complete normalization cases through allocating/destination/in-place
public APIs, and observes actual readonly flags, assembly paths/hashes and output
ownership. The auditor independently recomputes scalar normalization, checks
every value/setting, package ZIP/dependencies, actual restore selection and
build/package/cache/consumer byte identities. Full logs and private caches remain.

Stages inherit CPU0, with one .NET processor, shared compilation/node reuse
disabled, 300 seconds / 4 GiB sampled process-group RSS / 1 GiB available memory.
These small stages leave the active CPU2 numerical diagnostic and AMD VM alone.
Resource observations and exact PID/birth identities are retained; all owned
processes must be terminal before closure. No global environment or cache is
changed. Keep closure stdout outside the hashed artifact and verify the final
receipt after all writers exit.

Windows hardware does not exercise the AVX-512 path; the existing product
qualification supplies that proof on AMD. Core NuGet excludes Data/CLI and
native ORT. This is not audio packaging, a new default, or a timing result.
