# Remaining Whisper allocation after encoder reuse

The completed twenty-request conformance worker shows that decoder output
ownership is a larger remaining allocation issue than the logits copy alone.
This analysis reads its sealed result, models' configuration and exact source;
it runs no inference and does not qualify the separate endurance worker.

Across requests 2–20, public calls allocate **3,082,086,064 bytes**. Every one of
those requests allocates exactly **62,433,696 new pool bytes** in its first
decoder execution. The following required output shapes sum to that value:

| First-decoder output | Float payload bytes/request |
|---|---:|
| Eight encoder-attention key/value tensors, each 1×20×1500×64 | 61,440,000 |
| Eight initial decoder-attention tensors, each 1×20×4×64 | 163,840 |
| Four vocabulary rows, each 51,866 floats | 829,856 |
| Total | 62,433,696 |

Together these first-decoder pool allocations account for about **38.5%** of
the observed warm public allocation. They are outputs held by the generation
code, so increasing the existing cache of already-released arrays cannot by
itself make them reusable. Any future reuse must explicitly preserve the old
cross-attention values throughout the request and the previous self-attention
values until the next decoder call finishes. General graph outputs currently
remain valid after subsequent calls; that default ownership contract must hold.

`WhisperGeneration.DecodeCore` also makes a private logits copy on every step
before suppression. Its copy payload totals **149,374,080 bytes** across those
nineteen requests, or **4.85%** of their public allocation. A hypothetical single
four-row scratch array per request would remove **133,606,816 payload bytes**,
only **4.33%** of the observed total. That is a source-derived allocation estimate,
not an implemented saving or an end-to-end speed measurement. The modest share
does not make it the primary next memory mechanism.

The analysis also retains logical self-attention output sizes from all decode
steps. These are explicitly not attributed as fresh allocation: tensor views,
additional copies and release decisions need separate evidence. Array headers,
graph bookkeeping and all other allocations are outside the payload formulas.
Decoder telemetry reports only the final execution of each context in a public
request; its final past-decoder counter must not be summed as if it covered all
generated tokens.

The source is the exact private `18e10e3` archive used by the buffer-reuse
prototype. Completed worker result SHA-256 is
`58d8f24fe38d1a0eba0a110c5b580a4949737f115bb9394f9f9271a36c781be1`;
the bytes are copied and verified against the original conformance gate.
`artifacts/whisper-allocation-review-20260920` retains the full result, original
identity/gate snapshot, all twenty derived rows and every input hash.
Run `allocation_review.py` only against fresh output evidence; the completed
review must not be overwritten. No model, binary, source default, runtime
setting or acceptance threshold changes in this analysis.
