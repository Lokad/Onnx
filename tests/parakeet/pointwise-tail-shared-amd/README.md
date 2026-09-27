# Shared-model and e5 qualification for pointwise remainder sharing

Application closure `505da6ab` admits current Core `47984318` versus unchanged
candidate `7cac6788`: all 63 controls and 21 gates pass, with a 2.132562% matched
Parakeet gain. Model closure `f773277d` passes complete Parakeet correctness.
The rejected component screen and its runtime diagnosis remain unchanged.

Reuse Replay `a50d3e96`, original native arrays and existing models. Four original
processes run selected-shared, selected-e5, candidate-shared and candidate-e5.
Each product checks 166 arrays / 5,000,814 values, including all five e5 cases,
context reuse, memory policy, immutable inputs and independently held outputs.
Keep native scaled error <=1e-4, finite/shape checks and byte-identical candidate
versus current outputs. No consumer or product build occurs, and no Data assembly
is loaded. Compiled provenance reconciles all 3,983 original Core/Data methods;
one body changes, two internal helpers are added, and original flags/public
bindings remain exact.

Protocol, worker, checker and auditor match decoder-lstm-layout-shared-amd byte
for byte. The prepare function retains every original fixture; remote preparation
changes only the model/application namespaces. Seven prerequisite tests exercise
the actual admitted pair and rejection of unrelated changes or failed evidence.

Preserve CPU2 compute / CPU0 monitoring, 11 GiB RAM / 3 GiB tmpfs preflight,
8 GiB owned RSS, >=1 GiB RAM/tmpfs remaining and 900 seconds/job. Preserve the
original limits and supervisor; observe to terminal, collect once and audit once.
This is correctness, with no performance measurement or release promotion.

After freezing tools, prefix commands with `C:/Python313/python.exe -X utf8 -B`:

    -m unittest discover -s tests/parakeet/pointwise-tail-shared-amd -v
    tests/parakeet/pointwise-tail-shared-amd/run.py prepare
    tests/parakeet/pointwise-tail-shared-amd/run.py stage
    tests/parakeet/pointwise-tail-shared-amd/run.py launch
    tests/parakeet/pointwise-tail-shared-amd/run.py observe
    tests/parakeet/pointwise-tail-shared-amd/run.py collect
    tests/parakeet/pointwise-tail-shared-amd/audit.py

Local artifact: `artifacts/parakeet-pointwise-tail-shared-amd-20260927`.
VM: `/dev/shm/lokad-pwt-shared-20260927`. Pyannote, graph, portable narrow-width
coverage and actual-root/package qualification remain required afterward.
