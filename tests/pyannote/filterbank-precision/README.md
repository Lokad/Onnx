# Fixed-coefficient WeSpeaker precision experiment

This diagnostic tests three source-defined intermediate rounding groups and
their union across all21 existing frontend inputs and32 pipeline windows. It
keeps the captured managed Hamming/mel float coefficients, exact `.97f`, FFT,
framing and final float output type. It changes no product code or acceptance
threshold. The [ExecPlan](../../../.agent/m3-filterbank-precision-20260920.md)
defines the prospective arithmetic, complete-array checks and nomination rules.

Variants are Original; Frame (mean/DC/preemphasis/window in double); Spectrum
(FFT components/magnitude/power in double); Output (log/centering in double);
and All (those three groups together). Generated sources refuse product drift.
The original must reproduce every prior Windows feature bit. Seven complete
stages and final float output bytes are saved for every case/variant.

The new double references use the same managed tables. NumPy/OpenBLAS and
Torch/MKL implement the complete function independently, reusing the already
qualified reference algorithms. Old native-table double references and original
native features remain separate comparisons at their unchanged gates. The
managed and native Hamming tables differ, so their references are not relabelled.

From the repository root, after committing the new tools:

    C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-precision/prepare.py --artifact artifacts/wespeaker-precision-20260920
    C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-precision/run.py --artifact artifacts/wespeaker-precision-20260920
    C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-precision/audit_precision.py --artifact artifacts/wespeaker-precision-20260920

Build and all three computation workers run sequentially on CPU0 with one
numerical thread. Each has600second/2GiBRSS/1GiBavailable guards and4GiBavailable/
8GiBdisk preflight. Destinations refuse overwrite; all calls, failures, source,
input/coefficient/library identities and process resources are retained. This
Windows-only frontend diagnostic runs no recognizer, embedding or diarization
network and supplies no application latency or AMD arithmetic conclusion.
