# Separate the remaining Parakeet LSTM questions before another change

The retained original-decoder trace attributes **64.8072% of its sampled LSTM
intervals to `LstmProjectOrdered`**. The other **35.1928% remains unresolved**
inside the LSTM method. This read-only review reuses the qualified current product,
one original decoder fixture and all 1,024 post-warmup calls. It runs no inference,
builds no product and proposes no additional implementation while the independent
[packed-row application comparison](../decoder-packed-row-app-amd/README.md) runs.

| Named stack category | Reconstructed worker milliseconds |
| --- | ---: |
| LSTM ordered projections | 2,128.378332 |
| Other time under the LSTM method, unresolved | 1,155.791681 |
| Final vocabulary row-major kernel | 1,076.098787 |
| Other decoder work | 1,400.651526 |
| **Total inside observed decoder calls** | **5,760.920326** |

The original interval accounting conserves every reconstructed worker interval,
including intervals outside calls and on other threads. This review changes only
the stack-name categories and proves the same total, profiles and rounding
adjustments. These are sampled-thread estimates from an instrumented repeated
fixture, not direct component timers or full-corpus performance scores.

Almost all unresolved LSTM time has the generic `CPU_TIME` leaf; 1.280936 ms has
`UNMANAGED_CODE_TIME`. Neither label identifies sigmoid, tanh, allocation or a
particular native instruction. No per-call JIT-version association is available.
Do not label this remainder activation time or scale its proportions into a
predicted application saving.

The [original export inspection](../selected-profile-results/decoder-lstm-model-20260924.md)
identifies two one-step, batch-one LSTMs with hidden/input size 640. Each W/R
constant is `[1,2560,640]`, 6,553,600 bytes. The retained ORT node profiles show
X, bias, initial hidden state and initial cell state; W/R are absent from ordinary
profile inputs. Together with the source below, that supports prepacked weights.
It does not prove which native routine every LSTM invocation executed.

| Exact implementation difference | Lokad current source | Pinned ORT source |
| --- | --- | --- |
| Prepared weight representation | Owned `[input,gate]` transpose; projection advances 10,240 bytes between reduction rows | `TryPackWeights` calls `MlasGemmPackB` for W and R; `ComputeGemm` uses the packed-GEMM overload when available |
| Projection arithmetic | Four portable vector accumulators, separate multiply/add, increasing reduction order; independent input/recurrent output buffers | Input GEMM uses beta zero; recurrent GEMM accumulates into it with beta one |
| Default activations | Scalar delegates call `MathF.Exp` and `MathF.Tanh` for each hidden element | `ActivationFuncByName` selects bulk `MlasComputeLogistic` / `MlasComputeTanh`; the output merge also uses the bulk tanh helper |
| Constant lookup | Array/tensor identity, shape and length checks | Prepared-weight dispatch in `ComputeGemm` |

Lokad's lookup does **not** hash or transpose all weights on every call.
The already-shipped prepared-recurrence work must not be proposed again as a new
optimization. The recent Tensor sigmoid optimization also does not replace the
LSTM's separate scalar activation delegates.

The two projection-buffer accumulation strategies can round differently.
Changing prepared layout, adopting FMA/ORT accumulation, and replacing activation
arithmetic are separate mechanisms. Their combined implementation would obscure
the cause of any gain or numerical difference. The observed projection share
supports investigating that path first; any activation proposal still needs
attribution that distinguishes it from the unresolved remainder.

The older full-corpus node profile reports 2.314126 seconds managed versus
0.995040 seconds ORT for the two LSTMs, a 1.319086-second difference. Those are
dated profile observations, not this review's fresh matched score or a guaranteed
recoverable saving. The original native profile remains dated September 24.
The installed native GEMM sample previously identified a routine shared by several
decoder nodes; its global samples must not all be assigned to these LSTMs.

After the active application verdict, choose one implementation decision from
these distinctions. Reuse the original captured trajectories and exact graph
outputs. Request additional observation only if the missing dispatch, arithmetic
or attribution detail could change that decision. Keep the current candidate's
verdict separate; do not add another change to rescue its result.

Managed source: [recurrence](../../../src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs),
[ordered projection](../../../src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs),
[prepared ownership and lookup](../../../src/Lokad.Onnx/GraphLstmPacking.cs).
ORT source revision: `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`; copies of
`deep_cpu_lstm.cc`, `uni_directional_lstm.cc` and `rnn_helpers.cc` are retained in
`artifacts/parakeet-decoder-lstm-current-review-20260927/native-source`.

[All identities, interval totals and native shapes](observations-20260927.json).
Reproduce without inference:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-lstm-current-review/analyze.py

Publication is complete. Do not replay the original decoder capture.
