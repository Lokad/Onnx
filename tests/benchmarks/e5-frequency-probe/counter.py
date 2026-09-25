"""External perf interval capture with explicit monotonic epoch bounds."""
import decimal
import json
import os
from pathlib import Path
import select
import sys
import time

EVENTS = ['msr/aperf/', 'msr/mperf/', 'msr/tsc/', 'task-clock',
          'context-switches', 'cpu-migrations', 'page-faults']
UNCERTAINTY_NS = 20_000_000


def intervals(text):
    groups = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split(';')
        assert len(fields) >= 6, line
        stamp = decimal.Decimal(fields[0].strip())
        assert stamp.is_finite() and stamp >= 0
        ns = stamp * 1_000_000_000
        assert ns == int(ns)
        name = fields[3].strip()
        assert name in EVENTS, line
        raw = fields[1].strip()
        if raw == '<not counted>':
            count = None
        else:
            count = decimal.Decimal(raw)
            assert count.is_finite() and count >= 0, line
        runtime = decimal.Decimal(fields[4].strip())
        fraction = decimal.Decimal(fields[5].strip())
        assert runtime.is_finite() and runtime >= 0 and fraction.is_finite() and 0 <= fraction <= 100
        if not groups or groups[-1]['elapsed_ns'] != int(ns):
            if groups:
                assert list(groups[-1]['events']) == EVENTS
                assert int(ns) > groups[-1]['elapsed_ns']
            groups.append(dict(elapsed_ns=int(ns), events={}))
        row = groups[-1]
        assert name not in row['events']
        row['events'][name] = dict(count=None if count is None else str(count),
            unit=fields[2].strip(), runtime=str(runtime), running_percent=str(fraction), raw=line)
    assert groups and list(groups[-1]['events']) == EVENTS
    return groups


def epoch(anchors):
    assert len(anchors) == 2 and [a['command'] for a in anchors] == ['enable', 'disable']
    for a in anchors:
        assert a['before_ns'] <= a['ack_ns'] <= a['after_ns']
        assert 0 <= a['after_ns'] - a['before_ns'] <= UNCERTAINTY_NS
    low = max(a['before_ns'] - a['elapsed_ns'] for a in anchors)
    high = min(a['after_ns'] - a['elapsed_ns'] for a in anchors)
    assert low <= high and high - low <= UNCERTAINTY_NS
    return dict(lower_ns=low, upper_ns=high, uncertainty_ns=high-low)


class Counter:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir()
        self.output = self.folder/'intervals.csv'
        self.anchors = []
        for name in ['control', 'ack']:
            os.mkfifo(self.folder/name, 0o600)
        self.control = os.open(self.folder/'control', os.O_RDWR | os.O_NONBLOCK)
        self.ack = os.open(self.folder/'ack', os.O_RDWR | os.O_NONBLOCK)

    def command(self):
        return ['sudo', '-n', '/usr/bin/perf', 'stat', '-a', '-C', '2', '-D', '-1',
            '-I', '1000', '-x', ';', '-e', ','.join(EVENTS), '-o', str(self.output),
            '--control=fifo:'+str(self.folder/'control')+','+str(self.folder/'ack'),
            '--', sys.executable, '-B', str(Path(__file__).resolve()), 'waiter', str(self.folder)]

    def anchor(self, command):
        assert command == ('enable' if not self.anchors else 'disable')
        # Wait for perf's workload to exist, proving counter/control setup finished.
        deadline = time.monotonic()+10
        while not (self.folder/'ready.json').exists():
            assert time.monotonic() < deadline, 'Perf helper did not start'
            time.sleep(.001)
        old = self.output.read_text() if self.output.exists() else ''
        before = time.monotonic_ns()
        os.write(self.control, (command+'\n').encode())
        received = b''
        while received != b'ack\n':
            assert time.monotonic() < deadline, 'Perf acknowledgment missing'
            assert b'ack\n'.startswith(received)
            if select.select([self.ack], [], [], .001)[0]:
                received += os.read(self.ack, 4-len(received))
        ack = time.monotonic_ns()
        while True:
            assert time.monotonic() < deadline, 'Perf forced interval missing'
            text = self.output.read_text()
            assert text.startswith(old)
            try:
                fresh = intervals(text[len(old):])
            except (AssertionError, decimal.InvalidOperation):
                time.sleep(.0005)
                continue
            after = time.monotonic_ns()
            assert len(fresh) == 1, 'Control boundary mixed with another interval'
            item = dict(command=command, before_ns=before, ack_ns=ack, after_ns=after,
                elapsed_ns=fresh[0]['elapsed_ns'], output_before_bytes=len(old.encode()),
                output_after_bytes=len(text.encode()))
            assert after-before <= UNCERTAINTY_NS, item
            self.anchors.append(item)
            return item

    def stop(self):
        (self.folder/'stop').touch(exist_ok=False)

    def close(self):
        os.close(self.control)
        os.close(self.ack)


def waiter(folder):
    folder = Path(folder)
    (folder/'ready.json').write_text(json.dumps(dict(pid=os.getpid(), monotonic_ns=time.monotonic_ns())))
    start = time.monotonic()
    while not (folder/'stop').exists():
        if time.monotonic()-start >= 900:
            raise TimeoutError('Counter owner did not stop before deadline')
        time.sleep(.01)


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] == 'waiter'
    waiter(sys.argv[2])
