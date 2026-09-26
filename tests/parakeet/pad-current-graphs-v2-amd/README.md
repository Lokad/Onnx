# Restore the graph worker's required staging directory

The first current-padding graph campaign closed as a launch failure at
5745067d. Its unchanged worker starts processes with `cwd=BASE/source`, but
preparation omitted the existing `source/global.json` input and consequently
the directory. The first Popen raised FileNotFoundError. No model worker,
inference, resource sample or timing existed. Preserve all 155 collected files
and the original failure; do not alter or restart that campaign.

This successor restores the original pinned `source/global.json` input and
checks it exists before launch. It uses the exact original 72 workers, three
compiled consumers, native scripts, checks, scorers and full auditor from
pad-current-graphs-amd. Both products and all numerical, timing, repeatability,
ownership and resource gates stay identical. There is no product build or
performance variant. The failure receipt is a frozen prerequisite; its owner
must be terminal before staging.

The original protocol remains eight graph cases, 6,000 warmups for eight-token
e5, 1,200 for 30-token e5 and 600 otherwise, followed by 180 measurements per
timed process. Preserve all 73,512 calls, 8,640 measurements and 72 setup intervals.
Require native scaled error <= 1e-4, exact managed outputs, <= 5% regression and
<= 10% process disagreement. Keep CPU 2 compute, CPU 0 monitoring and all prior
11 GiB / 3 GiB preflight, 8 GiB RSS, 900-second job and four-hour campaign limits.

From repository root, prefix with `C:/Python313/python.exe -X utf8 -B`:

    -m unittest discover -s tests/parakeet/pad-current-graphs-v2-amd -p test_*.py
    tests/parakeet/pad-current-graphs-v2-amd/run.py prepare
    tests/parakeet/pad-current-graphs-v2-amd/run.py stage
    tests/parakeet/pad-current-graphs-v2-amd/run.py launch
    tests/parakeet/pad-current-graphs-v2-amd/run.py observe
    tests/parakeet/pad-current-graphs-v2-amd/run.py collect
    tests/parakeet/pad-current-graphs-v2-amd/audit.py

Only observe may repeat while active. Freeze tools at preparation, preserve any
failure, and collect/audit once after all owners are terminal. Audit stdout stays
outside the artifact folder. No release promotion follows from staging.

Local: `artifacts/parakeet-pad-current-graphs-v2-amd-20260926`.
VM: `/dev/shm/lokad-parakeet-pad-current-graphs-v2-20260926`.
