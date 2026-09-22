# Current Pyannote diagnostic on AMD

This frozen diagnostic refreshes attribution for selected Core208371f6/Data
b9358370 (M22), after the distinct M23 loop experiment failed its fixed gain
gate. It changes neither the product nor the matched ORT/application scoreboard.
Pyannote remains first, Parakeet second, Whisper deferred.

Build closure f2b4960a2ffd29987455a93a3db95ca2a7b3b1fba19005dced5afbe8e9cba6c3
qualifies the exact current product and SampledAudio consumer: all160 methods
compared,159 unchanged, only Main's two expected library hash operands change.
It uses SDK10.0.204/runtime10.0.8. Its predecessor build without global.json
stopped at SDK selection and remains preserved in its own namespace.

Preparation verifies the complete previous profile, current application and
consumer build closures. All original fixtures, models, native expected
outputs, parser and collector inputs retain their exact pins. The current
application manifest differs only in product hashes and source label.

The original bounded Linux capture worker is reused byte for byte. One control
and two sampled processes each run four warmups and twelve measured requests:
48 complete requests. Preserve the post-warmup release barrier, CPU2 targets,
CPU0 collectors,180second barrier,1.1second process-birth rounding allowance,
900second pairs,10GiB available/3GiB tmpfs initial preflight,8GiB owned RSS,
1GiB minimum available/tmpfs and1GiB output,2GiB experiment artifacts.
Every public output must equal the current application reference exactly.

Only after all capture owners are terminal and traces are collected, convert
each sampled trace into Speedscope and Chromium on AMD using the original
pinned exporter. Each of the four serial converters is CPU0, has900seconds,
8GiB available/3GiB tmpfs preflight,8GiB owned RSS,1GiB minimum available/tmpfs,
1GiB output per capture directory and2GiB total experiment artifacts. Record
every child PID/birth and every thread affinity; do not overlap model timings.
Collect only terminal owners; preserve input pins before and after conversion.

The independent audit reconciles all48 complete outputs,36 measured calls,
every resource sample, consumer equivalence, both exports, exact process and
thread ownership, no warmup contamination and three marker intervals per
fixture. Keep sampled thread weights, complete-request wall and process CPU
separate. Attribution alone never selects a faster product or changes an ORT
ratio. Use it to specify one distinct candidate with prospective gates.

From the repository root use C:/Python313/python.exe -X utf8 -B and the scripts
in this directory: test_selected_stacks.py, test_export_audit.py, prepare.py;
then transport.py stage, launch, observe, collect; then export.py launch,
observe, collect; finally audit.py. Refuse existing destinations. Failed
attempts remain immutable and require explicit successor directories.
Closure file keys are repository-relative. Models and tools already exist;
no downloads, product rebuilds or new VM-window permission are needed.

The initial local preparation stopped before staging because it assumed an
application-reference field named passed. The reference uses the original
public result schema. This explicit v2 successor invokes the full existing
public validator and preserves the failed tools and partial payload unchanged.
