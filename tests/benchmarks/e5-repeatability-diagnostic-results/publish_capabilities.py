"""Publish the exact available counter scope for the next bounded observation."""
import json
from pathlib import Path
from probe_counters import ROOT, pin, read

OUT = Path(__file__).resolve().parent


def main():
    reports = {}
    for name, folder in [('hardware', 'e5-hardware-counter-capabilities-20260925'),
                         ('frequency', 'e5-frequency-counter-capabilities-20260925')]:
        base = ROOT/'artifacts'/folder
        proof = read(base/'closed.json')
        assert proof['passed'] and proof['observations'] == pin(base/'observations.json')
        value = read(base/'observations.json')
        assert value['inference_calls'] == 0 and not value['system_settings_changed']
        reports[name] = dict(closure=pin(base/'closed.json'), observations=value)
    hardware = reports['hardware']['observations']
    frequency = reports['frequency']['observations']
    assert 'cpu' not in hardware['sources']
    assert all('<not supported>' in row['stderr'] for row in hardware['commands'][1:])
    assert frequency['code'] == 0
    counters = {}
    for line in frequency['stderr'].splitlines():
        fields = line.split(';')
        assert len(fields) == 7 and fields[0].isdigit() and float(fields[4]) == 100
        counters[fields[2]] = int(fields[0])
    assert set(counters) == {'msr/aperf/', 'msr/mperf/', 'msr/tsc/'}
    assert all(value > 0 for value in counters.values())
    value = dict(reports=reports, counters=counters,
        probe_aperf_over_mperf=counters['msr/aperf/']/counters['msr/mperf/'],
        inference_calls=0, system_settings_changed=False, diagnosis_complete=False,
        source=pin(Path(__file__)))
    document = OUT/'counter-capabilities-20260925.md'
    data = OUT/'counter-capabilities-20260925.json'
    assert not document.exists() and not data.exists()
    data.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    document.write_text('''# Available VM counters for the next e5 observation

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
''', encoding='utf8')
    print(json.dumps(dict(passed=True, frequency_counters=counters, inference_calls=0)))


if __name__ == '__main__':
    main()
