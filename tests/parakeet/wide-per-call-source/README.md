# Wide Parakeet projection dispatch candidate

Start from selected source94a550de and the complete root420file inventory.
Require both closed current AMD profiles to attribute at least20% of their
complete corpus to the AVX2 packed consumers; observed shares are about44.2%.
Prepare only an isolated copy. Change two existing consumer guards in
Tensor<T>.RunFloatMatMulKernel: allow the existing AVX512 row-sharing helper
when both reduction/output axes are at least1024, retaining the broad opt-in.

The existing packing eligibility, scratch rental/return, clearing, packed layout,
row grouping, column refusal, odd-row fallback and all raw arithmetic remain.
Parakeet retained packing budgets remain256MiB/64MiB. This does not enable the
broad dynamic switch by default or alter e5 attention geometries.

Run C:/Python313/python.exe -X utf8 -B prepare.py from the repository root.
Refuse existing artifact parakeet-wide-per-call-source-20260923. A prepared
source is not a numerical or speed qualification. Normal actual-DLL build and
full numerical/component/application admission must precede integration.
