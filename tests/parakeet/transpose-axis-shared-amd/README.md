# Shared-model and e5 qualification for tiled axis movement

Application closure 2e4741a6 admits this exact candidate: all 63 controls and
21 gates pass, with 2.420895% matched gain and candidate/ORT 1.108577118.
Complete-model closure 8568a229 and focused closure c0c063bb pass. Parity remains
open. Current Core 4e97e2ae and candidate c471f5d1 differ only in TransposeInto;
all 3,287 other Core methods, 697 Data methods, flags and public bindings remain
unchanged. No product or consumer is rebuilt.

Reuse original Replay a50d3e96 and every native fixture. Four serial processes
run selected-shared, selected-e5, candidate-shared and candidate-e5. Each product
checks 166 arrays / 5,000,814 values, all five e5 shapes, repeated context reuse,
memory policies, immutable inputs and independently held outputs. Require scaled
native error <=1e-4, finite values, exact shapes and bit-exact current/candidate
arrays. No Data assembly is loaded and no performance is scored in this lane.

The worker, protocol, checks and auditor remain byte-identical to
decoder-lstm-layout-shared-amd. The preparation function retains every fixture;
remote preparation changes only model/application namespaces. The local adapter
binds the actual one-method scope, model proof and admitted application, and
rejects failed controls, wrong consumers or unrelated method changes.

Keep CPU2 work / CPU0 monitoring, 11 GiB RAM / 3 GiB tmpfs preflight, 8 GiB RSS,
1 GiB RAM/tmpfs remaining, 1 GiB output, 2 GiB stage and 900 seconds/job. Reuse
retained assets. Run all seven prerequisite tests and freeze the tools before
preparation. From repository root, prefix with `C:/Python313/python.exe -X utf8 -B`
and run `tests/parakeet/transpose-axis-shared-amd/run.py` with prepare, stage,
launch and observe. Follow the same owner to terminal, collect once and run
audit.py once. Never replay a completed stage or overwrite failed evidence.

Artifacts: parakeet-transpose-axis-shared-amd-20260928.
VM: /dev/shm/lokad-transpose-axis-shared-20260928. Pyannote, graph performance and
actual-root/package qualification remain required afterward. Root product and
BENCHMARK.md remain unchanged until release qualification completes.
