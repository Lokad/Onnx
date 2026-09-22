# Direct final-row output for Pyannote convolution

This isolated consumer retains the selected product DLL and original
three/two-row assignment. Its new three-row method starts accumulators at zero,
performs the original full reduction, adds optional bias, and stores into the
final row stride. The final two/four rows retain the original two-row kernel
and scalar bias/copy pass. Non-admitted shapes retain the original route.
No reduction blocking, shared MatMul change or product promotion is included.

The closed convolution census supplies all 22 geometries, four actual row
strides and 33 first/last tile offsets. Qualification has 2,882 cases per mode:
finite boundaries, real geometry, signed zeros and non-finite values, with
and without bias. Complete candidate/baseline bits must match, including NaNs;
the independent scalar oracle checks exact finite/infinity bits and NaN
classification. Every output padding value, guard and input is checked.
Normal and AVX2-disabled modes execute on Windows and AMD. Finite/zero hashes
must agree across platforms and modes; NaN payloads remain exact against each
mode's selected baseline, without imposing a new cross-mode payload contract.

Four fresh normal AMD timing processes run baseline/candidate/candidate/baseline.
Clearing, packing, multiplication and bias/copy are inside timing. Patch
expansion is common and outside this synthetic tile scope. Each process
conditions all 22 shapes for one second each and warms each again before six
measured blocks. Timing uses each shape's last actual tile offset with bias,
as present in all 36 model convolutions. All samples are retained.

Fixed screening gates: every repeated-role process ratio <=1.10, equal-shape
geometric mean candidate/baseline <=0.95, and every shape ratio <=1.05.
Passing permits full product/model qualification only; it is not a full-request
speedup, ORT comparison or calibrated confidence interval.

From repository root, prefix these commands with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/direct-output/prepare.py
    tests/pyannote/direct-output/transport.py stage
    tests/pyannote/direct-output/transport.py launch
    tests/pyannote/direct-output/transport.py observe
    tests/pyannote/direct-output/transport.py collect
    tests/pyannote/direct-output/audit.py

Do not repeat completed stages; observe an existing PID/birth owner until
terminal and then collect once. Artifacts use `artifacts/pyannote-direct-output-20260922`
and `/dev/shm/lokad-pyannote-direct-output-20260922` on the exclusive AMD VM.
Pin payload and installed runtime files before and after execution. Targets
inherit CPU 2 before CLR; monitor uses CPU 0 and checks every observed native
thread. Each process is bounded by 900 seconds and 2 GiB RSS, retaining 1 GiB
available memory and tmpfs free. Preflight needs 8 GiB available and 3 GiB
tmpfs free; total artifacts <=1 GiB. Local build/check bounds are unchanged.
Preserve failures and use explicit successors; never relax gates after results.
