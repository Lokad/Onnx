# Draft Parakeet last-axis padding candidate

Not prepared, built, measured or selected. Finish M43/M45 qualification first.
`prepare.py` requires the complete M43 root proof, all 421 exact parent inputs
and committed product source before creating an isolated 423-file snapshot.

The candidate inserts one guarded call in PadCore after the existing fill and
empty-output return. A new private helper copies contiguous source rows for
nonnegative last-axis padding with zero outer padding. Source materialization,
fill, crop/reflection fallback, validation and result ownership remain intact.
Zero-width input returns the already-filled destination without division.
The helper's source file sorts after existing CPUExecutionProvider files, so
the normal build can require exact existing compiler-generated method names.

Six draft public-operator tests cover all four supported element types, an
independent coordinate oracle, exceptional float bits, empty shapes, layouts,
views, cropping/reflection and result ownership. These templates have not yet
been compiled or run; they are added only to the isolated candidate's suite.

All 48 actual encoder Pad nodes resolve to this family. The current profile
assigns PadCore about 3.8% of request samples; this is no measured candidate gain.
See the [source review](../current-profile-results/memory-source-review-20260923.md).

After the parent qualifies, use `C:/Python313/python.exe -X utf8 -B` with
`prepare.py`, then qualify a normal AMD build, independent bitwise numerics,
complete public Pad cost and complete model/application performance. No root
integration or BENCHMARK.md update follows source preparation alone.
