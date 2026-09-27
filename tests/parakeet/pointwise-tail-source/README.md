# One eight-row column-remainder candidate

The closed pointwise observation assigns 95.77% of time to arithmetic. Source and
natural shape variation identify the two-accumulator remainder loops. This fixed
candidate shares each remainder vector across eight rows, retaining AVX2,
ascending reduction, existing packing and FMA operand order. Masked columns
retain separate multiply/add. Full 32-column statements are unchanged.

`prepare.py` copies the 443 qualified root inputs, patches only the remainder
portion of `mm_unsafe_vectorized_intrinsics_2x4packed_bump`, and adds two private
helpers. The source patch reverses exactly. Eight-row helpers apply only at the
existing large-work boundary, M>=64 and N>=64; remaining row pairs and all other
paths preserve their original code. No runtime flag, row-size sweep or additional
optimization is included. The root product remains unchanged.

From the repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/pointwise-tail-source/prepare.py

Prepare once after source review. Numerical and code-generation qualification
must precede timing. Neither this snapshot nor a passing raw contract promotes
the candidate to a release.
