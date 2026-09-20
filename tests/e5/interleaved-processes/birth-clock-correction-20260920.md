# Preserve process identity without mixing Linux timestamp domains

The first complete AMD cohort exposed an offline-auditor error before any
candidate decision. The producer and full experiment continue unchanged.
No timing sample, numerical/resource threshold or control criterion is altered.

The frozen auditor compared a worker's `psutil.create_time()` with the cohort's
`time.time()` launch timestamp. On this VM, psutil constructs creation time from
the process start counter plus the **integer-second** `btime` in `/proc/stat`.
Those epoch-looking values therefore differ from the wall-clock timestamps.
For the first worker, the recorded creation time is `1789919120.82`, while
the cohort begins at `1789919121.6240542` and launch is recorded at
`1789919121.631317`. Its supervisor's process creation time is `1789919120.22`.
Every later sample and live query matches the same original worker PID/birth.

The separate `audit_birth_clock.py` changes exactly one assertion in the
frozen `telemetry` function: a worker birth must follow its supervisor birth,
which uses the same time base. Recorded wall-clock launch/finish ordering is
still checked separately. All PID/birth equality, uniqueness, command sequence,
affinity, suspended state, inactive CPU, memory, deadline and final-termination
checks remain. Stopwatch inference durations are unrelated to this correction.

The installed VM psutil source, its hash/version, original live-state snapshot,
all first-cohort resource records and a source diff are retained at
`artifacts/e5-interleaved-processes-v3-20260920/birth-clock-review`.
Four tests demonstrate the original refusal on this actual complete cohort,
the corrected acceptance, and continued refusal of an older worker birth,
reversed wall-clock ordering or an inactive process observed running. The test
uses an explicitly labeled single-complete-cohort view; the original full
phase is still running and is not declared complete.

Use `audit_birth_clock.py` for the final independent audit and
`close_phase_birth_clock.py` for phase closure. Both delegate to the original
frozen tools, preserve their files and verify the correction's receipt.
No worker, successful stage or previously closed inference is replayed.
