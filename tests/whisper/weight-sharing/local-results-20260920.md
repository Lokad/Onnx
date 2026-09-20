# Private Whisper decoder-weight sharing: local proof

The prototype shares **635,187,200 bytes** of identical initializer storage
between the first and past decoders inside one transcriber. An actual model
preparation check reduces unique initializer backing storage from
**1,985,006,252 to 1,349,819,052 bytes**, with **587 to 499 arrays**. These are
referenced initializer payloads, not total managed heap or process RSS savings.

All initializer names, tensor names, types, full shapes and full-content hashes
remain identical before and after sharing, including prepared folded and packed
entries. Node names/operators/input/output bindings and retained packed-byte
totals also match. The independent serialized-model census predicts exactly the
observed shared payload after applying the helper's FP32/4 KiB eligibility rule.
It preserves independent tensor wrappers and each graph's own preparation state.

The complete backend suite passes **3,101 tests**, with **93 skipped**. Its twelve
new cases check backing-array identity, separate metadata, signed-zero and NaN
bits, shape/layout/size exclusions, observable inputs/outputs, prepared MatMul
rebuilding, exact outputs, held-output lifetime and input preservation. General
graph contracts and production source are unchanged; the helper is used only by
the separate private Whisper prototype before execution contexts are created.

The inspection runs on Windows/.NET 10.0.12, CPU 2, with no runtime overrides or
native ORT. It performs model import/preparation and byte hashing, **no inference**.
All 13 resource samples pass a 120-second, 8 GiB RSS and 4 GiB
available-memory guard. Peak observed RSS is 3,959,349,248 bytes; the
original process identity is terminal. That one-process peak is not a before/after
RSS comparison. No explicit garbage collection is used by the .NET consumer.

The helper/product build has zero warnings/errors. The test build retains four
nullable warnings in unresolved-binding fixtures. The inspection build retains
two platform-analysis warnings for its Windows affinity check. Its first build
failed on generic inference for three `MemoryMarshal.TryGetArray` calls; explicit
type parameters fix the consumer in a new `inspection-v2` directory. An earlier
launcher import found psutil absent from global Python and now uses the already
installed isolated psutil 7.0.0. No global installation or model rerun occurred.

The private source inherits the independently tested buffer-reuse prototype and
adds this distinct storage change. Its AMD application, numerical and repeated
request resource qualification remains pending. The existing buffer-reuse
endurance process is a different immutable artifact and is not modified by this
work. Existing memory/numerical failures remain unchanged.

[Summary observations](local-observations-20260920.json) record exact counters,
binary result identity and resource bounds. Full source pins, source diff, both
inspection build attempts, successful tests, every model-initializer hash and
resource samples are under `artifacts/whisper-weight-sharing-20260920`.
