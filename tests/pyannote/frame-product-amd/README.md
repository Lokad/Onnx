# Public frontend qualification on AMD

[Completed results](results-20260920.md) qualify product `1d10d22` against
independent mathematical references, while retaining direct-native failures.
All 99 affected tests and all 53 inputs pass. Connected natural-meeting replay
with this changed frontend is a separate pending step.

The successful preparation is `prepare_v4.py`; it archives exact product source
with explicit LF settings, includes the solution marker and builds the CLI before
the affected tests. Earlier preparation scripts and the first failed AMD test
attempt are preserved. Do not rerun these single-use writers into closed artifacts.

`remote_v2.py` supervises each fresh AMD stage with a 600-second, 2-GiB group-RSS
bound, CPU 2 affinity and memory/workspace checks. `vm.py` launches once and collects
only after every recorded process birth is terminal. The completed collections
contain every source, build log, TRX, coefficient table, feature array and binary.
Work ran in a unique memory-backed directory because the VM root disk was nearly
full. No earlier evidence or model was deleted.

The captured AMD tables differ from Windows at six entries. `prepare_references.py`
binds those exact tables and all original PCM inputs to the previously qualified
NumPy/OpenBLAS and Torch/MKL algorithms. `run_references.py` runs both routes under
finite local process guards, saving all seven complete stages for every input.
`audit.py` checks all 742 reference arrays, independent scalar/DFT probes, 212 full
product/reference comparisons, native diagnostics, source blobs, test counters,
duplicate inputs and process/resource records. `close.py` freezes the complete
evidence and preserves reporting failures without repeating inference.

The final closure binds 3,623 artifact files and four reports/tools. Independent
verification checks all of them, 65 external pins, 15 numerical libraries and
every original process identity. Artifacts and source identities are listed in
the report and its [complete observations](observations-20260920.json).
