# Run the M78 phase and complete-projection diagnosis

The source preparation at ba592033 is already complete. Do not rerun it or edit
its files. The remaining adapter uses those exact 22 Data sources, the unchanged
PhaseProbe and SampledAudio 38ab5c7e. Original M78 Core 49901366 and Data 01e9e784
form the control. Only diagnostic Data and the inventory bridge are built.

M78's graph comparison is closed: all numerical and 24 repeatability checks pass,
but 8-token e5 regresses 7.7654%, failing the 5% release limit. Preserve that
verdict. This diagnosis profiles the isolated candidate; it cannot promote it or
change BENCHMARK.md. Requiring release admission before diagnosing an isolated
candidate would prevent useful Parakeet work for an unrelated e5 failure.

Preparation requires the closed direct-release Parakeet comparison, the original
admitted M78 application comparison, exact model/compiled qualification and the
closed graph evidence, including its failed verdict. It checks every closed
input, consistent product identities and all 433 isolated source files. Numerical
checks must pass; any timing failures remain visible in diagnostic_context.
Pyannote timing, root integration and release promotion remain pending separately.
No profile VM work may overlap the active application campaign. Local preparation
also freezes every adapter tool. The older README is frozen source-preparation
evidence; this runbook describes the current campaign order.

Use Python 3.13 -X utf8 -B from the repository root. Run unittest discovery and
consumer_scope.py first. After the direct comparison closes, execute run.py prepare and
run.py stage once. Then run.py launch build, observe build while its owner is live,
collect build after all exact PID/birth owners are terminal, and review_build.py.
Keep review stdout outside the campaign directory. A timeout is not terminal.

The six build jobs are SDK version, Data restore/build, bridge restore/build and
compiled inventory. SDK is 10.0.204; every dotnet command uses --tl:off. Keep the
existing offline feed and raw pinned assembly references. Require no build
warnings. Both runtimes retain every original dependency byte; only observed
Data changes. All 164 runner methods remain identical. Of 697 original Data
methods, only Execute may differ; its scope must recover the original body,
branches, locals and disposal behavior. Its 50 added helpers and method flags
must match the retained certified observer. The packed-weight constructor
remains byte-equivalent to measured M78 in compiled inventory.

Only after the build review is transferred, run.py launch capture. Three fresh
processes run control, phase and wall, with the original request loop and resource
supervisor. Each runs all 20 clips, one warmup and three measured passes: 240
requests. The only capture-loop change supplies the explicit measured Core hash
already supported by the reviewed runner. CPU 2 executes; CPU 0 monitors.
Build bounds remain 2 GiB available/1 GiB tmpfs before each command, 3 GiB RSS and
180 seconds/command. Capture bounds remain 11 GiB available/2 GiB tmpfs before
each process, 12 GiB RSS and 900 seconds/process. Both retain 1 GiB remaining
memory/tmpfs and at most 512 MiB campaign output. No profile is a score.

Observe capture while live; collect capture once terminal, then audit.py once.
Keep audit stdout outside the campaign directory. Every raw request, result,
phase/node clock, owner and resource record remains. The original attribution
checker and its tests are unchanged. Additionally require every complete public
result to equal admitted M78, identical phase/wall graph metadata, the same
runner/Core and the reviewed packed-weight constructor. Report both overhead
steps without subtraction. Account for frontend, encoder, decoder and all work
outside graph calls before selecting any narrower group.

After audit, compare.py matches all 217 constant projections against the dated
native graph/profile, including 48 fused half-scales and all 265 managed nodes.
Other nodes and dynamic products remain in full raw analysis. That comparison
does not refresh ORT timings or BENCHMARK.md. Identify the largest measured
remaining phase/group first; reject a projection-focused hypothesis if other
work dominates or if the applicable loop already runs at comparable cost.

Local campaign: artifacts/parakeet-packed-final-row-profile-amd-20260925.
VM campaign: /dev/shm/lokad-parakeet-packed-final-row-profile-20260925.
Source receipt: artifacts/parakeet-packed-final-row-profile-source-20260925/prepared.json.
Preserve failures and never repeat an unchanged completed workload to obtain a
favorable observation. Preparation, build, review, capture and audit are distinct;
passing local checker tests does not establish that the new observer executes.
