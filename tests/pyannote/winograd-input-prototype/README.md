# Eight-tile Winograd input transform

M31 changes only TransformWinogradInput in the isolated M30 prototype. It
hoists coordinate and padding-mask computation outside the channel loop and
uses eight-lane AVX2 gathers and explicit ordered add/subtract transforms.
Every lane, including padding and incomplete batches, is initialized. All
other arithmetic, guards and selected direct sources remain exact.

The complete 1152 raw and 87 captured cases per ISA, 21 refusals and 24
alias/extent cases remain unchanged. All output hashes, native error statistics
and ownership records must equal M30 exactly; the original native bound stays
1e-4. This is a numerical campaign with no product dispatch or timing claim.

From the repository root use C:/Python313/python.exe -X utf8 -B followed by
`tests/pyannote/winograd-input-prototype/run.py prepare`, then `stage`, `launch`,
`observe`, terminal `collect` and `audit.py`. Before preparation run
`-m unittest discover -s tests/pyannote/winograd-input-prototype -p test_audit.py`.
Do not overwrite or relaunch a frozen attempt. The ordinary seven-job AMD
build and numerical protocol retains SDK10.0.204/runtime10.0.8, CPU2 workers,
CPU0 monitor, 10GiB build /12GiB numerical preflight and 3GiB tmpfs preflight.
Live bounds are 8GiB owned RSS, 1GiB available/tmpfs, 900s/job, 1GiB output and
2GiB artifacts. The only ISA override disables AVX512 for 256-bit correctness.

M30's complete-call screen was rejected at 30.82% slower; no favorable subset
is selected. A subsequent M31 screen must retain all 87 captured calls, at
least 10% aggregate gain, no form more than 5% slower, every repeatability
control and strict process separation. Full application/ORT gates are unchanged.
