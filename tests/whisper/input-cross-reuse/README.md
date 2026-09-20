# Whisper encoder comparison with one reused context

This preserves the full scope of the [original diagnostic](../input-cross/README.md):
twenty recordings plus the first repeat, both encoders on both saved feature
arrays, all 42 exact baseline bridges and all 84 full encoder outputs. The first
attempt stopped at its fixed 8 GiB memory guard and remains a closed failure.

The managed worker now retains one `GraphExecution`, resetting it before and
after every call. Its two first returned tensors remain held and checked through
the entire sequence. No product arithmetic, model, input, threshold, runtime
override or resource limit changes. Passive allocation, collection and pool
counters provide additional evidence without requesting garbage collection.
This lifecycle differs from the public transcriber, so this is a numerical
localization experiment, not public-pipeline or performance qualification.

Build this `Probe.csproj` with the original frozen product directory, outputting
to the new artifact `artifacts/whisper-input-cross-reuse-20260920/bin`. Run this
directory's `test_audit.py` and `prepare.py --artifact <new-directory>`.
Use the original `../input-cross/run.py --artifact <new-directory>` with
`--engine managed`, then `--engine native` only after the managed supervisor and
all children are verified terminal. All original memory/time/affinity guards apply.

Run this directory's `audit.py --artifact <new-directory> --output <artifact>/audit.json`
and `close.py --artifact <new-directory>`. The new metadata auditor delegates all
full-array, input, baseline, held/repeat, decomposition and resource checks to the
original independent auditor. Preserve all numerical failures and every attempted
run; do not replay an existing output destination.
