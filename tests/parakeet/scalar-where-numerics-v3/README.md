# Uniform-mask Where numerical qualification

The V3 source6971271e normal-build scope audit closes at8f5958cb. Compare its actual
Core70275a50 against selected Core672e5f30, using the same independent consumer.
No root source changes. The failed V1 scope build remains retained atfaa41f5e.

Freeze131 cases before execution: eight original capture fixtures and24 derived
true/mixed masks;25 small broadcast cases;ten fallback-rank/non-scalar cases;
four empty cases;24 layout/ownership cases; eight other dtypes; six invalid
shapes including unselected operands; three nulls; eleven float scalar bit
patterns including both zero signs, infinities, subnormals and NaN payloads.
Masks change at the first/last element or alternate. The layout cases include
memory offsets, semantic Dense subclasses, reversed one-dimensional tensors,
strided slices, a broadcast view and overlapping input storage. This census
does not establish arbitrary multi-dimensional reversed-layout semantics.

Every valid call must match an independent coordinate oracle and leave all
input backing arrays/guards unchanged. Mutate inputs, call again and verify
the first output stays exact; mutate the first output and verify all inputs
and the second output remain unchanged. Match every captured output byte.
Candidate float cases also invoke the private assembly's internal helper through
reflection and check declared admission/refusal, oracle bits and untouched inputs.
Non-float calls use the unchanged public fallback. Invalid calls must retain
the selected exception type. Compare every selected/candidate result hash after
collection. The CPU provider's dtype/error suite remains required for release.

Nine serial jobs: SDK, consumer restore/build, four numerical processes for both
products in normal and AVX512-disabled modes, and two normal-mode code-generation
processes exercising32 captured/derived cases81 times each. No measured latency.
Review the complete emitted Where/Try bodies before any performance screen.

Use run.py prepare,stage,launch,observe,collect, then audit.py. Tools and census
must be frozen before stage. Existing offline feed and32 captured arrays are
reused; fixture files are hardlinked only after validating the terminal capture
owner and hashes. Preflight12GiB available/3GiB tmpfs;8GiB RSS,1GiB remaining
memory/tmpfs,1GiB output/job,2GiB/campaign,900s/job,four hours/campaign.
CPU2 computes,CPU0 monitors. Preserve failed receipts; no root integration until
all later component, application and cross-model release gates pass.

V2 passes the canonical123-case census but fails raw masks[0,2]and[0,255]
in separate diagnostic e83bf8c6. V3 searches bytes by zero/nonzero truth.
Append all eight raw masks to the unchanged123cases; compare direct selected
bits, helper admissions, mutated masks/inputs and independent NumPy uint32
selection for every added case. Expect48candidate admissions per mode.
The separate code-generation capture remains32cases, never a timing result.
