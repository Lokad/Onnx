# Build and qualify the single dense-slice conversion

The isolated source adds one `TensorSlice<T>.ToDenseTensor` override. It calls
the existing bounded contiguous-copy helper, then the original base fallback.
Every original method body, implementation flag and helper guard must remain
exact. The declared-member inventory intentionally gains the override; its
base definition and effective inherited signature must stay the same.

The preceding copy-only comparison remains unadmitted because 13 helper
per-clip repeatability controls failed. Stable generic-copy cost and complete
source/counter evidence justify this isolated prototype. Neither that result
nor these correctness checks establishes an application speedup.

Freeze these tools, then run `run.py prepare`, `stage`, `launch build`, observe
that exact owner and collect only after terminal. `review_build.py` checks the
complete instruction inventory and unchanged warning baseline. Only its passed
review permits `launch capture`: the complete tensor suite in ordinary and
AVX512-disabled modes. Expect 395 passing tests per mode: 368 existing, 26 new
conversion contracts and one identity/override contract. Preserve every result.

Use Python 3.13 `-X utf8 -B` for local orchestration. All .NET work uses the
exclusive AMD VM, CPU2, CPU0 monitoring, SDK10.0.204/runtime10.0.8, offline restore
and `--tl:off`. The two existing CS8604 warnings in Zzz.WideProjectionEntry.cs
are disclosed and must not change. After correctness qualification, preserve
the existing full model, application and release gates before integration.
