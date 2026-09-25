# Available VM counters for the next e5 observation

Ordinary CPU cycles, retired instructions, cache references and cache misses
report **not supported** on this VM. The CPU performance-counter device is absent.
Software task-clock, context switches, migrations and page faults work.

The separately advertised APERF, MPERF and TSC counters return positive readings
with 100% reported enabled time. APERF counts actual-performance cycles and
MPERF supplies a reference-rate count; their ratio can help investigate clock-rate
variation. These are virtual-machine observations: the probe does not establish
physical-host frequency accuracy, nor explain any prior model timing.

Both probes ran only small bounded Python arithmetic workloads after all model
owners were terminal. No model inference or system-setting change occurred.

The next investigation should observe only the failed long-input workload,
reusing its existing product, consumer and all 780 calls. First establish an
auditable alignment between counter intervals and the existing call boundaries.
Record software counters and frequency ratios; retain every interval and call.
The prediction is that a frequency-driven change would associate slower blocks
with a lower APERF/MPERF ratio. A stable ratio would reject that explanation in
the observed run. No frequency change or new warmup policy is authorized by
this capability check, and the original failed release controls remain retained.

[Probe outputs, counts, commands and closure hashes](counter-capabilities-20260925.json)
and [runtime interpretation](interpretation-20260925.md) provide the evidence.
