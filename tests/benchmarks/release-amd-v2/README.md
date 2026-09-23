# Current release graph comparisons on AMD

Freeze this protocol before collecting any clocks. Use the selected actual Core
521bae17, qualified as equivalent to all methods in root product94a550de, and
Microsoft ORT1.29.0 on the exclusive AMD EPYC9V74 CPU2. No implementation switch,
tiering override, profiler, calibration, trimming or adaptive retry is allowed.
These are current-release baselines, not candidate admission experiments.

The eight cases are e5 at8,30,30-padded-to128,128,512tokens, DINOv3 ViT-S/16
224x224, ResNet50 feature export224x224, and GPT-2 four-token prefill with empty
past state. Use the already qualified immutable input/output fixtures. No download.
For each case run separate verification workers for current and ORT before any
timed worker. Then use four fresh processes per case: current,ORT,ORT,current.
Each performs60fixed warmups and60measured requests; retain all clocks and setup
durations. Each request is one complete forward call returning owned float arrays.
Managed timing includes Reset, Execute and output materialization. ORT timing
includes session.run returning all arrays. Model load/preparation, input creation,
reference checking, hashing and reporting are outside the timer. Both use default
memory behavior; ORT enables all graph optimizations and one intra/inter thread.

Check every output on every call against unchanged native references at
abs(actual-reference)/max(1,abs(reference))<=1e-4. Check names, shapes, finiteness,
input immutability, deterministic outputs and retained output ownership. Save all
first outputs and compare managed/native workers independently after collection.
No latency is publishable until these checks pass. Repeated process means for
each role/case must have max/min<=1.10. A failed control retains all observations
but withholds a qualified ratio for that case; no unchanged retry follows.

Run on CPU2 with CPU0 monitor and no overlapping benchmark owners. Retain exact
PID/birth identities, every sampled thread affinity, RSS, memory, disk and elapsed
limit. Preserve existing limits:12GiB numeric preflight,10GiB build preflight,
3GiB tmpfs preflight,8GiB RSS,1GiB minimum available/tmpfs,900seconds/job,
1GiB job output and2GiB campaign artifacts. Pin actual loaded product, native
binaries, fixture/model bytes and harness sources before launch. Retain failures.

From repository root run C:/Python313/python.exe -X utf8 -B with run.py
prepare,stage,launch,observe,collect, then audit.py. Existing artifact and remote
namespaces are refused. Collection requires every owned identity terminal.
Whisper optimization remains deferred; DINOv2 remains explicitly excluded by the
repository's numerical divergence registry. Neither receives recycled old timings.

The original campaign stopped at consumer compilation: ITensor exposes Dims,
not the typed tensor Dimensions property. No numerical or timing worker ran.
Correction v2 changes only the two shape-check property accesses and campaign
identity. The complete original build failure remains under release-amd and its
artifact closure; all numerical, timing, resource and reporting rules are exact.
