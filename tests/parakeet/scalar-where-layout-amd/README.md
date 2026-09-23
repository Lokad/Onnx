# Selected Parakeet Where layout census

Use the unchanged selected Core `672e5f30` and Data `065b7a7f`. Two retained
frontend outputs from `english-16k` and `jfk-48k-stereo` produce 74 and 138
encoder frames. Their full outputs must match the selected release bitwise
and the retained native reference at the existing 1e-4 scaled bound.

An existing debug logger callback captures all 73 Where inputs immediately
before each node, and its output before the next node runs. Node sequence and
identity must reconcile completely. No product patch, graph output tap, graph
policy change, model copy or performance measurement is used. Debug logging
adds overhead; its clocks cannot become a benchmark.

Export exactly four fixed nodes per recording: `/Where`, layer 0 attention
`Where`, attention `Where_1`, and convolution `Where`. Eight fixtures contain
32 arrays; four complete encoder outputs make 36 arrays total, capped at 32 MiB.
The independent auditor broadcasts and selects the exported integer bit patterns
to verify every output. It records observed eligibility, including refusals;
the initial candidate scope is a hypothesis, not a required census outcome.
Feed immutability and held outputs are checked across context reset/reuse.

Run `C:/Python313/python.exe -X utf8 -B run.py prepare`, stage, launch, observe,
collect, then `audit.py`. SDK 10.0.204/runtime 10.0.8; all builds/inference on AMD,
CPU2, monitor CPU0, unchanged 12 GiB memory/3 GiB tmpfs preflight, 8 GiB RSS,
1 GiB minimum available memory/tmpfs, 900 seconds/job and four hours total.
Keep all resource samples with gaps below ten seconds. Existing namespaces are
refused; source snapshots, consumer, selected references and models are pinned.
