# Rational sigmoid: focused correctness and generated-code checks pass

The fixed ORT-derived rational sigmoid passes all **12 focused tests normally
and 12 with hardware intrinsics disabled**. Each mode checks 2,048,769 sweep
values. Maximum absolute float error is **1.7881393e-7**, below the unchanged
1e-6 bound; disabled-intrinsics sweep error is zero. Special values, saturation
boundaries, tails, layouts, immutable inputs, independently owned outputs,
double behavior, invalid-input handling and actual product identities pass.

This is an isolated candidate. No complete-model or transcription performance
claim follows from these contracts. The release source and BENCHMARK.md retain
the qualified padding product.

| Product | Core SHA-256 | Data SHA-256 |
|---|---|---|
| Qualified root | `8bb22038d0b4c09b56b2cdae06c49c165b8e646bc73ca28ad400f4ace0bfc659` | `d02dbf550d7a6ea0ddf24985ffff7b86db135035ce7fab31f4bab063e0090620` |
| Rational candidate | `946ddfb66492c48a0fc6078ecbe1957ac494ff5d9ff0be42259d70e66e3b1f24` | `dbe959361209bbc20db9bd566f037f58e01b9cc40d807c5745dbe2f4e1c1aca6` |

The compiled inventory verifies 3,281 unchanged existing Core methods, only the
public Sigmoid method changed, and exactly one private helper added. Its
NoInlining flag keeps vector arithmetic separate from the public scalar loop.
All 697 Data methods, existing method flags, public surfaces and assembly
attributes match. Only the two existing CS8604 warning sites appear.

The helper uses Microsoft's fixed coefficients and order from ORT revision
2e2543fbe9fae542f921d47a72d21d5a4ef0b710, retaining the MIT notice. It computes
sigmoid without input multiplication. Public output allocation and the graph
structure are unchanged; the helper handles its own scalar remainder, while
the original zero-based scalar fallback remains separate.

The actual normal process confirms Vector<float>.Count = 8. Its optimized
Tier1-OSR helper is 638 bytes, with a 53-instruction vector loop containing nine
FMAs and division using 256-bit operands. The vector body has no exponential
call or integer conversion. Disassembly retains every emitted version. The
normal test process emitted only Tier0 for the public wrapper; these observations
do not identify its eventual optimized fallback or associate a tier with a clock.
The disabled-intrinsics process never emits the vector helper.

Both contract processes and the build supervisor are terminal. Resource limits
pass; peak contract process-tree RSS is 341,430,272 bytes. Logging affected only
the correctness processes. Ordinary-runtime timing uses a separate common consumer
and retains the original 46-case screen, rounds, process order and every gate.

[Candidate preparation](../rational-sigmoid-source/prepare.py),
[build and contract protocol](../rational-sigmoid-build/README.md),
[performance protocol](../rational-sigmoid-screen/README.md),
[underlying diagnosis](../sigmoid-execution-results/diagnosis-20260927.md).

Source receipt: `1ca5891c2223e3fd9fa5c65c12c45e3bbeb96f45fc5903d11ccd950895577eb7`.
Build review: `2619dee6c8f64ea91578d8c63665878f8ed1181da8cbf464dd0514b4af2092e7`.
Contract closure: `3301ae58b42e1fd51435f54191cf4bba42c6cbf8e2ad6bf4279af67ae5d3d89a`.
Complete numerical evidence and disassembly:
`artifacts/parakeet-rational-sigmoid-build-amd-20260927`.
