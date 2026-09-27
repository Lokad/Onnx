# Original shared-model and e5 checks for the unchanged LSTM layout

Execution is conditional on the actual independent Parakeet application verdict.
Bind its successful admitted closure before freezing or preparing. A synthetic
unit-test admission cannot permit execution; a rejected application stops here.

Compare qualified Core `0d224bcf` with candidate `ad97b4ad`, reusing Replay
`a50d3e96`, original native arrays and existing models. No Data assembly is loaded,
and no consumer or product is rebuilt. The completed compatibility chain binds
all 3,981 original Core/Data methods through the actual root: only LSTM preparation
and prepared dispatch change, with two internal reader/getter methods added.
All original public bindings, flags and assembly metadata remain exact.

Four original processes run selected-shared, selected-e5, candidate-shared and
candidate-e5. Each product checks 166 arrays / 5,000,814 values across its pair of
jobs, including all five e5 shape/padding cases, context reuse, memory policy,
immutable inputs and held outputs. Native scaled error remains <=1e-4 for all
finite values. Candidate arrays must also be byte-identical to current.

Protocol, worker, checker and auditor match decoder-packed-row-shared-amd byte
for byte; the prepare function retains all original fixtures. Remote preparation
differs only in its model/application namespace. Prerequisite binding describes
this exact pair and requires all 63 application controls and 21 gates. Keep the
failed component screen and its runtime diagnosis explicit; this lane neither
rescales old clocks nor establishes universal fallback equivalence.

Preserve 11 GiB RAM / 3 GiB tmpfs preflight, 8 GiB owned RSS, 900 seconds/job,
>=1 GiB RAM/tmpfs remaining, CPU2 compute and CPU0 monitoring. No VM work overlaps
the active application trial. Check disk before collecting; reuse canonical
models and retained binaries. This is correctness, not a performance score.

After application admission, prefix with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/decoder-lstm-layout-shared-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/decoder-lstm-layout-shared-amd -v
    tests/parakeet/decoder-lstm-layout-shared-amd/run.py prepare
    tests/parakeet/decoder-lstm-layout-shared-amd/run.py stage
    tests/parakeet/decoder-lstm-layout-shared-amd/run.py launch
    tests/parakeet/decoder-lstm-layout-shared-amd/run.py observe
    tests/parakeet/decoder-lstm-layout-shared-amd/run.py collect
    tests/parakeet/decoder-lstm-layout-shared-amd/audit.py

Freeze tools first, observe the same owner to terminal, collect once and keep
audit output outside the artifact. Never replay completed phases.
Local artifacts/parakeet-decoder-lstm-layout-shared-amd-20260927;
VM /dev/shm/lokad-lstmlayout-shared-20260927. Pyannote, graph and actual-root/package
qualification remain required afterward.
