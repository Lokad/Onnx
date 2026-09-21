"""Identities and owned-process tools for a normal source/package integration."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-portable-integration-20260922'
ACCEPTED = ROOT / 'artifacts/pyannote-sparse-mel-20260921'
APPLICATION = ROOT / 'artifacts/pyannote-sparse-mel-applications-20260921'
INVENTORY = ROOT / 'artifacts/pyannote-integration-review-20260921'
GUARD = ROOT / 'artifacts/pyannote-lstm-panel-admission-completion-v3-20260921/candidate-source'
DENSE = ROOT / 'artifacts/pyannote-convolution-portable-applications-20260921/application-runtime/Lokad.Onnx.Data.dll'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('portable_integration_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil

RECEIPTS = [
    (ACCEPTED / 'focused-closed.json', '35cba7a867ee2faff0fb8c3b94431c6f3ffa390648242188da2cd6515856cb6b'),
    (APPLICATION / 'closed.json', 'f687396b6a6966b80e41f100dbe3bdddd0c682c8d63bfc658744ce05f5d79a3d'),
    (ROOT / 'artifacts/pyannote-sparse-mel-comparison-20260921/closed.json', 'ea57759fc9bd39976014520cbb0d2abf88300b6b14fcf54811b88773e24c90aa'),
    (ROOT / 'artifacts/pyannote-sparse-mel-comparison-finish-20260921/closed.json', 'c2ffb923788aceab97361b470346c7ba206f8afae0bc0fdba586ca4cdb34fec5'),
    (INVENTORY / 'closed.json', 'c3e2b4defdb097d5988c6b71c7e5b9b5bbe65941fed30670028c66ae88f22a53'),
    (ROOT / 'artifacts/pyannote-lstm-panel-admission-completion-v4-20260921/closed.json', '2923462378198220a5284a4308e6cfe723c80efa3f5656ad11ba3b107786ce2d'),
]


def rel(path):
    return path.relative_to(ROOT).as_posix()


def prerequisites():
    files = {rel(MONITOR): pin(MONITOR)}
    for path, sha in RECEIPTS:
        assert pin(path)['sha256'] == sha
        proof = read(path)
        assert proof['passed']
        verify(proof['files'])
        files[rel(path)] = pin(path)
        identities = proof.get('identities', proof.get('terminal_identities', []))
        identities += [dict(pid=int(pid), birth=birth) for pid, birth in proof.get('identities_verified_absent', {}).items()]
        for identity in identities:
            terminal(identity)
        for name, wanted in proof.get('external_files', {}).items():
            assert pin(Path(name)) == wanted
    assert pin(APPLICATION / 'application-runtime/Lokad.Onnx.dll')['sha256'] == '5c0ae2aa7c3cce58f3ffcb190df451e053a449a3e0dbc920d7b6d0b2bc66020c'
    assert pin(APPLICATION / 'application-runtime/Lokad.Onnx.Data.dll')['sha256'] == 'e9e4c28e2f7277ea226556f692d95de3d9d4a5f94eed79a9235268137dcd4775'
    assert pin(DENSE)['sha256'] == '1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662'
    return files


def suite(name, passed, skipped):
    import collections
    import xml.etree.ElementTree as ET
    path = BASE / 'test-results' / (name + '.trx')
    tree = ET.parse(path)
    counters = tree.find('.//{*}Counters').attrib
    outcomes = collections.Counter(r.attrib['outcome'] for r in tree.findall('.//{*}UnitTestResult'))
    assert outcomes == dict(Passed=passed, **({'NotExecuted': skipped} if skipped else {})), outcomes
    assert int(counters['passed']) == passed and int(counters['notExecuted']) == skipped and int(counters['failed']) == 0
    return dict(name=name, counters=counters, outcomes=dict(outcomes), trx=pin(path))
