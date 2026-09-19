# Packed projection overwrite experiment — September 19, 2026

The overwrite prototype passes arithmetic and storage checks, but does not clear
its prospective performance screen. Keep it outside the product and do not start
a full-model comparison for this result. The 128-row projection bank improves
1.23%, below the required 2%; the thirty-row bank has large swings in unchanged
controls and an aggregate 3.35% regression against the original. The eight-row
and 512-row improvements remain useful observations, not model speedup claims.

ORT 1.23.2 at `a83fc4d58cb48eb68890dd689f94f28288cf2278` uses `ZeroMode`
in `onnxruntime/core/mlas/lib/sgemm.cpp` when beta is zero, then accumulates
subsequent reduction blocks. The reviewed voice branch's `be97ba9` cohort also
offers overwrite behavior. Lokad's current packed projection path clears its
destination, then loads those zeros into the accumulators. This experiment
adapts only initial overwrite; it preserves the current reduction order,
packing, tile geometry, and row partition.

Five modes share the exact same packed buffer for every matrix:

| Mode | Computation | Clear before each call |
|---|---|---|
| 0 | Original accumulating composer and leaves | Yes |
| 1 | Exact duplicate of mode 0 | Yes |
| 2 | Copied accumulating composer and leaves | Yes |
| 3 | Copied overwrite composer and leaves | Yes |
| 4 | Same overwrite code as mode 3 | No |

The source audit verifies that only 60 full-panel accumulator initializations
change from destination loads to positive-zero vectors. All twelve/eight-row
bulk and three/two-row remainder arithmetic and stores remain unchanged.
Accepted column counts are positive multiples of 32; copied partial-column
tails retain their accumulating semantics and are unreachable behind that
guard. This is not a replacement for the public accumulating methods.

Ten fresh processes cover five cyclic mode orders and their reversals, rotated
by shape. Each isolated projection has eight warmups and seven measured batches;
batch sizes are 64/64/16/4 for 8/30/128/512 rows. The bank has 72 independent
matrices, representing twelve layers with four width projections, one expansion
and one contraction each: 84,934,656 packed bytes, exceeding the 32-MiB L3.
Each bank mode has two warmups and seven measured passes. Allocation and packing
are outside repeat-use timing and recorded separately. Complete-call timings
include each mode's required clear. Activation dependencies are not modeled.

All runs use AMD EPYC 9V74, CPU 2, .NET 10.0.8, normal tiering and GC, no forced
collection and no removed samples. The supervisor runs on CPU 0. The screening
rule was fixed before timing: at least 2% mean improvement at both 30 and 128
rows against modes 0 and 2, wins in at least eight of ten workers for each,
gains exceeding duplicate/generated-control movement, and no aggregate
regression over 2% at 8 or 512 rows. This is a screening rule, not statistical
qualification.

| Rows | Original mean ms | No-clear overwrite mean ms | Duplicate / original | Generated accumulation / original | Overwrite with clear / original | No-clear overwrite / original | No-clear overwrite / generated accumulation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3.918233 | 3.764210 | 0.998434 | 1.001823 | 0.987471 | 0.960691 | 0.958942 |
| 30 | 12.477280 | 12.895854 | 1.186670 | 1.150006 | 1.173415 | 1.033547 | 0.898732 |
| 128 | 48.838183 | 48.236044 | 0.997643 | 0.999607 | 0.989679 | 0.987671 | 0.988059 |
| 512 | 193.841511 | 191.164951 | 1.000884 | 1.000649 | 0.988194 | 0.986192 | 0.985552 |

Ratios divide means of all retained samples. No-clear overwrite beats both
accumulating controls in all ten workers at 8, 128 and 512 rows. At 30 rows it
beats the original in seven workers and the generated control in nine. Its
worker ratios against the original range from 0.9692 to 1.4298. Every worker
has one thirty-row mode block with samples above 15 ms, reaching 26.2134 ms;
other blocks are generally around 12–13 ms. These slow blocks occur across
modes 1–4. In seven workers they are the second mode block and in three the
fifth. Their cause is not established. No GC collections occurred in the
recorded warmup/measurement intervals. The apparent 10.13% gain against generated
accumulation therefore does not establish a thirty-row speedup.

| Isolated rows × reduction × columns | No-clear overwrite / original | No-clear overwrite / generated accumulation | Duplicate / original |
|---|---:|---:|---:|
| 8 × 384 × 384 | 0.984660 | 0.989544 | 0.998245 |
| 8 × 384 × 1536 | 0.985225 | 0.983529 | 0.999700 |
| 8 × 1536 × 384 | 0.997452 | 0.996554 | 0.999581 |
| 30 × 384 × 384 | 0.988895 | 0.988852 | 1.000336 |
| 30 × 384 × 1536 | 0.995089 | 0.989262 | 1.003017 |
| 30 × 1536 × 384 | 1.008879 | 0.990312 | 1.008492 |
| 128 × 384 × 384 | 0.984878 | 0.984000 | 1.001426 |
| 128 × 384 × 1536 | 0.981648 | 0.979791 | 1.000715 |
| 128 × 1536 × 384 | 0.993181 | 0.994232 | 0.999377 |
| 512 × 384 × 384 | 0.981052 | 0.980526 | 1.000006 |
| 512 × 384 × 1536 | 1.018781 | 0.976351 | 1.045752 |
| 512 × 1536 × 384 | 0.994561 | 0.994042 | 1.055907 |

Removing the clear alone, comparing identical overwrite code in modes 4 and 3,
gives bank ratios 0.972880/0.880803/0.997971/0.997974. The thirty-row ratio has
the same control problem. Most of the stable 128/512-row improvement comes
from changing accumulator initialization, with about 0.2% additionally observed
from omitting the clear.

Every worker passes 256 finite/exceptional geometry cases and eight unsupported
geometries across all three implementations. Checks cover full output bits,
independent scalar ordered-FMA coordinates, dirty and repeatedly overwritten
destinations, original/generated nonzero and repeated accumulation, signed
zero, NaNs/infinities, sliced input/output guards, unchanged source/packed
values and stable shared addresses. The local AVX2 host verifies unsupported
refusal and allocation/packing contracts; arithmetic checks execute on AMD.

A separate eleventh process captures generated code. Original and generated
accumulation bulk methods have identical instruction text and sizes: 907 bytes
for eight rows and 1,135 for twelve. Overwrite versions shrink to 733 and 890
bytes, respectively. Their initial 16/24 vector loads become zero instructions;
the reduction FMA sequences and store counts remain intact. No bulk vector
stack accesses or reduction-loop calls occur. The twelve-row loop retains
scalar stack loads: four in accumulation, five in overwrite, including the
panel pointer. Thus the change also affects register allocation. Three/two-row
remainders show instrumented Tier0 and Tier1-OSR versions in this diagnostic;
that observation alone does not attribute the thirty-row swings to compilation.

Independent audits validate the complete archive, source/binary identities,
orders, all 600 isolated records, 200 bank records and 5,600 samples, output
hashes, actual shared buffer addresses and process accounting. Eight corrupted
evidence cases are refused; all sixteen shape schedules are balanced. Peak
sampled process-group RSS is 193,486,848 bytes. Maximum observed foreign CPU
fraction is 0.00137455; snapshots cannot capture every short-lived process.

The frozen probe DLL is
`dbebbaabe94209a9a5fa98eb9721f5dc5abd76e7852c8d7378960dacc446f5c2`.
It references qualified core
`7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9`;
core execution source is unchanged between c6bf781 and d0fbe49. Bundle SHA256 is
`995b397d58fae8d063fe180427db5f49714534fdc034a46688afd0a57a912473`;
collected archive SHA256 is
`75c63a398650c2f876aabd3f1eb13579e215ca31f61f98fae527a6d00fa1d4f1`;
timing audit receipt SHA256 is
`fbe7573d758a93a650e548babdf228a48627492b96bd996c9fe2d6bedac02b94`.
The exact prospective plan is retained with its pre-run SHA256
`ff958e18cb86353ec0faa3309bc54b6c60853fd3f3954325dc17660220f2be38`.

All source, setup costs, addresses, raw samples, disassembly and receipts remain
under `artifacts/packed-overwrite-20260919`. Supervisor 290201 and all eleven
children are terminal; collection and timing/codegen audits are closed.
No product code, defaults, numerical tolerance or e5 score changed. This closes
the bounded prototype; it does not establish that overwrite is universally
unhelpful. Any investigation of the newly observed phase-related timing swings
must use a distinct diagnostic and retain these original results.
