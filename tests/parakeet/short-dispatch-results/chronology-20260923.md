# Rejected M41 screen: chronology for the next diagnostic

M41 remains rejected: M167/reduction1024/columns4096 is9.18%slower, and the
current M48/1024/1024 repeatability ratio is1.36270, exceeding1.20. No clock is
excluded, no gate changes and no application campaign follows. The observed
69.24%short-case reduction is an unqualified component observation.

Inspection of every clock in ten-call blocks reveals transitions worth tracing.
For M167/1024/4096, current runs begin around18.3ms and settle around17.4ms near
the end of the60warmups. Candidate runs remain around18.3ms, with repeatable
spikes in measured blocks60–69 and90–99. The failing current M48control includes
a6.95ms block at70–79 in one process, versus roughly2.5–2.7ms otherwise.

These patterns do not establish whether JIT tier changes, garbage collection,
allocation state or another factor caused either failure. The separate codegen
capture proves emitted bodies, not which body executed at every scored call.
A diagnostic that correlates CLR compilation/GC events with call boundaries can
answer that missing question. It cannot admit or rescore M41.

[Every ten-call block for both cases and all four processes](chronology-20260923.json)
derives from the complete unchanged clocks in
[the rejected screen](screen-20260923.md). Every warmup and measurement remains
in its original role; no trimmed mean or replacement score is produced.
