# Parakeet: actual projection routes narrow the next experiment

The qualified release remains **64.705703 s versus ORT 39.636305 s**, ratio
**1.632486**. This investigation changes no product or benchmark result.

Both diagnostic applications passed their original public/native-reference,
input and output-ownership checks: 160 requests, 17,360 observed projections,
20 clips and 19 encoded lengths. Core f95a13c5 is unchanged. All scratch/copy
counters agree with the source routes, with no unexplained calls.

## Where the remaining projection difference sits

Each projection is joined to the exact installed ORT graph's corresponding
call, clip and actual input shape. Complete groups include the 48 scales that
ORT fuses. Clocks below reuse the earlier qualified, lighter Lokad profile and
the retained ORT profile. The route metadata comes from the new capture of the
same Core and inputs; these are separate captures, not simultaneous observations.
The differences are diagnostic attribution, not promised application savings.

| Observed route | Calls per corpus | Lokad profile (s) | ORT profile (s) | Difference (s) |
|---|---:|---:|---:|---:|
| Retained weight, preparation accepted | 436 | 2.581348 | 2.294022 | 0.287325 |
| Retained weight, declined for row count | 304 | 2.331866 | 1.517573 | 0.814293 |
| Weight not retained | 3,600 | 32.571013 | 25.630488 | 6.940525 |
| **All 217 projections** | **4,340** | **37.484226** | **29.442083** | **8.042143** |

The 256 MiB encoder cap retains 37 weights. Every accepted prepared call requests
zero packing scratch. Every other call requests exactly `4*K*N` bytes, including
calls that have a retained mapping but fail the row-count guard. The total is
39.313 GB of scratch requests per corpus, of which 2.068 GB is associated with
declined retained weights. These are rental sizes, not physical memory traffic.
The arithmetic leaf was not sampled per node; kernel names remain source
predictions. The previous exact ORT native-kernel proof remains applicable.

The odd-row restriction is real, but matching ORT on its affected calls would
close only 0.814 diagnostic seconds, about 1.26% of current application latency.
It is a smaller lead than the uncached projections and does not independently
justify another broad kernel change or a packing-budget sweep.

## The positional projections are the sharper lead

The 24 `self_attn/linear_pos/MatMul` nodes cost **5.382415 s versus 2.644351 s**:
2.738063 seconds of the projection difference. Their weights are `[1024,1024]`,
but their input has **2T-1 rows**, whereas the other projections use T rows.
There are 38 actual row counts across the 19 encoded lengths. This distinction
matters both to copying and to the odd-row guard.

Both original and optimized graphs give all 24 nodes the same input:
`/pos_enc/Slice_output_0`. It slices a float `[1,9999,1024]` constant along axis 1,
step 1, from `5000-T` to `5000+T-1`. The installed ORT revision's Slice allocates
its output and copies contiguous regions once; its 24 consumers read that output.

Lokad returns a `TensorSlice<float>` view. Before multiplication,
`RequireBatchOperand` calls `RequireContiguous`, which calls `ToDenseTensor`.
`TensorSlice` inherits the base implementation that iterates coordinates and
copies each element. Counters confirm **480 input materializations and
524,353,536 copied bytes per corpus**. Every positional call copies exactly its
input length; the other 193 constant projections report no input copies.

An existing bounded contiguous-copy helper in `TensorSlice` handles this slice
geometry, but only `Reshape` calls it. The next single experiment is to quantify
these materializations against that existing helper on the exact positional
constant and all actual lengths. Only if that cost supports a useful application
gain should the helper be connected to dense conversion, preserving independent
ownership and the generic fallback. Copy time has not yet been isolated; the
2.738-second group difference also includes packing and arithmetic.

## Observation limits and recovery

The observer's control took 65.012628 s and its enabled run 89.951990 s, a
38.36% increase including observation and process variation. Nothing is
subtracted. Its complete projection intervals total 37.440597 s, close to the
lighter profile's 37.484226 s, but this is not a proof of zero observer effect.
The lighter profile supplies the ranking above; all heavier clocks remain.

The original control completed. The next process was refused before launch by
the unchanged memory preflight. A separate owner ran only that missing process,
on the same reviewed binaries, after memory recovered. Both owners are terminal;
the failure, successful control and recovery are retained. No successful
inference or build was repeated. All 1,267 resource samples passed, with peak
owned RSS 9,684,365,312 bytes. Repository plus artifacts remains about 48.91 GB.

The first analysis assumed all matrix rows equalled encoded frames. Its refusal
is retained; correction accounts explicitly for all 1,920 positional observations
and matches their actual ORT shapes. No model execution was repeated to fix analysis.

[Every route, shape, matched group and receipt](projection-routes-20260924.json),
[route analysis](review_projection_routes.py),
[shared-slice/source proof](projection-copy-20260924.json),
[copy-path review](review_projection_copy.py),
[compiled observer scope](projection-build-20260924.md),
[exact ORT native-kernel investigation](../ort-diagnosis-results/ort-kernels-20260924.md).
