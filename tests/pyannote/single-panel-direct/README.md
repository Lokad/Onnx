# Reuse convolution patches with one panel

For 1–32 columns, the accepted `MathOps.PackPanelsB` layout equals its input.
This separate prototype skips that copy in the previously qualified direct-output
component. Wider shapes retain packing. Arithmetic kernels and all 3,266 cases
remain unchanged. The preceding application's failed 3% gate stays failed.

From repository root, run `C:/Python313/python.exe -X utf8 -B` with `prepare.py`,
then `audit.py` after successful termination. Output goes to
`artifacts/pyannote-single-panel-direct-20260922`; existing output is refused.

The independent layout checker covers 1,400 cases in each normal/hardware-disabled
mode, including 660 identity cases and 296 wider finite counterexamples. Both
normal and forced-scalar-tail components run the complete 3,266-case schedule.
Every input bit and guard is checked, including NaN payloads and signed zeros.

Workers inherit CPU2; the monitor uses CPU0. Preparation requires 8 GiB available
memory and each worker stays below 4 GiB RSS and 900 seconds, with 1 GiB memory,
20 GiB disk and 1 GiB output guards. No VM work or performance measurement is
launched here. Product scratch ownership and complete model/application gates
remain separate requirements before integration.
