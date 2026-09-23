# M47 isolated padding-dispatch build and targeted tests

Not prepared or run. Uses qualified selected root 94a550de and the
isolated 422-input source from `../pad-dispatch-source/prepare.py`.

Eight fixed jobs: SDK 10.0.204, normal CLI restore/build, backend restore/build,
full Core/Data metadata inspection, six targeted public Pad tests, then those
six tests with AVX512 disabled. All existing 3,179 Core and 697 Data methods
must stay exact except four call targets in public Pad; original PadCore is
entirely unchanged and only one private dispatcher may be added. Public
surface, method flags and all generated names must remain exact. A separate
helper review is still required after the actual build. The inventory verifies
public Pad's complete body, permitting only those four resolved call operands;
all original instructions, offsets, branches, locals and exceptions remain exact.

The tests use an independent coordinate oracle across four types, exceptional
float bits, ranks/empty shapes, layout/view materialization, cropping/reflection
and ownership. They do not replace complete native/public model conformance or
the performance screen. They must run against the exact inspected candidate
Core/Data binaries; backend output identities are verified before and after.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`,
`launch`, `observe`, `collect`, then `audit.py`. Freeze before transfer; preserve
failures and refuse existing destinations. Normal .NET 10.0.8 on AMD CPU 2,
monitoring CPU 0, with only the named second test job disabling AVX512. Run SDK
commands under source/ containing global.json and use `--tl:off`.

Preflight: 10 GiB available / 3 GiB tmpfs; each job below 8 GiB RSS and 900
seconds, with at least 1 GiB available/tmpfs, 1 GiB output and 2 GiB campaign
files. Maximum campaign four hours. Reuse the selected root's offline feed and runtime binaries, plus the already
qualified M43 metadata inspector (including method flags). The inspector's
correctness is independent of the rejected product's speed verdict. No model copies, inference,
score, root edit or BENCHMARK.md update occurs in this build lane.
