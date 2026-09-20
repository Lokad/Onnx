# Exact fingerprint cache: product correctness

The default-off `LOKAD_ONNX_FINGERPRINT_STRINGS` switch reuses immutable prepared
string hash transitions. The standalone component experiment nominated it;
this separate replay exercises the actual product, including graph mutation,
context ownership and complete neural outputs. It supplies no latency ratio.

`prepare.py --artifact <new-directory>` requires committed sources, archives
the core, Data, test projects and every first-party C# file, and builds that
extracted source with SDK10.0.204. The same core DLL must appear in both test
outputs and the replay output. Existing models and native references are
verified by complete SHA256 identities and are never overwritten or downloaded.

Eight sequential AMD workers run full backend/tensor suites, e5 and shared
models with the switch disabled and enabled. Each child inherits CPU2 before
CLR startup; the supervisor uses CPU0. Limits are600seconds,12GiB process-group
RSS and1GiB available memory per worker. All process births and resource samples
are retained. Failed attempts remain evidence and are not silently replaced.

The e5 replay covers8/30/padded128/128/512 tokens under Default and Memory,
through both facade and explicit contexts. Each combination executes twice,
then after a harmless node-name mutation. Checks require exact cached/original
fingerprints, the intended cache state, unchanged inputs and held outputs, and
complete native scaled error<=1e-4. Missing-input failures also preserve held
outputs. DINOv3, ResNet50 and GPT-2 replay all106 existing reference arrays,
including carried decoder state. On/off output bytes must match exactly.

`audit.py --payload <collected> --assets <repository> --label amd --output
<new-file>` independently reads every float, all TRX results, frozen identities
and resource telemetry. `collect.py` first proves every original process birth
terminal and checks an exact archive inventory. Successful output writers are
single-use. The switch stays off pending a separately declared complete-model
comparison; correctness qualification does not repair earlier failed timing
calibration or close unrelated audio numerical gaps.
