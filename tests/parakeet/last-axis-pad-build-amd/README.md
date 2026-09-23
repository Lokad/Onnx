# Draft M46 padding build and targeted tests

Not prepared or run. Requires fully qualified, committed M43 root and the
isolated 423-input source from `../last-axis-pad-source/prepare.py`.

Eight fixed jobs: SDK 10.0.204, normal CLI restore/build, backend restore/build,
full Core/Data metadata inspection, six targeted public Pad tests, then those
six tests with AVX512 disabled. All existing 3,181 Core and 697 Data methods
must stay exact except PadCore; only one private helper may be added. Public
surface, method flags and all generated names must remain exact. A separate
composition review of PadCore is still required after the actual build.

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
files. Maximum campaign four hours. Reuse the closed parent's offline feed,
metadata inspector and immutable runtime binaries. No model copies, inference,
score, root edit or BENCHMARK.md update occurs in this build lane.
