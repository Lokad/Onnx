"""Test-only portability successor; product and NuGet bytes remain fixed."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-portable-integration-tests-20260922'
PRIOR = ROOT / 'artifacts/pyannote-portable-integration-20260922'
COMPLETE = ROOT / 'artifacts/pyannote-portable-integration-completion-20260922'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('portable_tests_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path):
    return path.relative_to(ROOT).as_posix()


def read_suite(name, passed, skipped):
    import collections
    import xml.etree.ElementTree as ET
    path = BASE / 'test-results' / (name + '.trx')
    tree = ET.parse(path)
    counters = tree.find('.//{*}Counters').attrib
    outcomes = collections.Counter(r.attrib['outcome'] for r in tree.findall('.//{*}UnitTestResult'))
    assert outcomes == dict(Passed=passed, **({'NotExecuted': skipped} if skipped else {})), outcomes
    assert int(counters['passed']) == int(counters['executed']) == passed
    assert int(counters['total']) == passed + skipped and int(counters['failed']) == 0
    return dict(name=name, counters=counters, outcomes=dict(outcomes), trx=pin(path))
