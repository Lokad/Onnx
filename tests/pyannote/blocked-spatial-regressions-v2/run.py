"""Use normal project test invocation, retaining the direct-DLL launch failure."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PREVIOUS = ROOT/'tests/pyannote/blocked-spatial-regressions'
sys.path.insert(0, str(PREVIOUS))
import run as original

FAILED = original.BASE
BASE = ROOT/'artifacts/pyannote-blocked-spatial-regressions-v2-20260922'
original.BASE = BASE; original.monitor.BASE = BASE; original.TOOLS = TOOLS
pin, read, save, verify, terminal = original.pin, original.read, original.save, original.verify, original.terminal
old_prior = original.prior
old_worker = original.monitor.worker


def close_failure():
    assert not (FAILED/'failure-closed.json').exists()
    value = read(FAILED/'inputs.json'); verify(value['files'])
    state = read(FAILED/'controller.json'); assert state['complete'] and state['code'] == 1
    row, = state['runs']; assert row['name'] == 'backend-existing' and row['complete'] and row['code'] == 1
    assert not list((FAILED/'test-results').glob('*.trx'))
    assert (FAILED/'logs/backend-existing.log').read_text().strip() == 'The following arguments have been ignored : "--no-build --no-restore"'
    identities = [state['supervisor']]+[dict(pid=int(p), birth=b) for p, b in row['members'].items()]
    for identity in identities: terminal(identity)
    samples = [json.loads(s) for s in (FAILED/'logs/backend-existing.samples.jsonl').read_text().splitlines()]
    assert len(samples) == row['samples'] == 2 and max(s['rss'] for s in samples) == row['peak_rss']
    assert row['preflight']['available'] >= 12*1024**3 and row['seconds'] < 900
    for s in samples:
        assert s['seconds'] < 900 and s['rss'] < 8*1024**3 and s['available'] >= 1024**3
        assert s['disk'] >= 20*1024**3 and s['output_bytes'] <= 1024**3
        assert s['rss'] == sum(m['rss'] for m in s['members'])
        for m in s['members']: assert m['affinity'] == [2] and row['members'][str(m['pid'])] == m['birth']
    files = {p.relative_to(FAILED).as_posix(): pin(p) for p in FAILED.rglob('*') if p.is_file()}
    save(FAILED/'failure-closed.json', dict(passed=False, retained_failure=True, files=files, identities=identities,
        regression_results_produced=False, resources_passed=True, local_inputs=value['files']))


def prior():
    value = old_prior()
    failure = read(FAILED/'failure-closed.json'); assert not failure['passed'] and failure['retained_failure']
    for name, wanted in failure['files'].items(): assert pin(FAILED/name) == wanted, name
    for identity in failure['identities']: terminal(identity)
    return value


def worker(controller, state_path, name, command, cwd, expected, preflight_gib, rss_gib, seconds, allow_children, output):
    kind = dict((n, k) for n, k, _, _ in original.SUITES)[name]
    assert command[:2] == ['dotnet', 'test']
    project = original.PRODUCT/f'source/tests/Lokad.Onnx.{kind}.Tests/Lokad.Onnx.{kind}.Tests.csproj'
    # --no-build/--no-restore keep the exact already-qualified normal product.
    args = ['dotnet', 'test', project, '-c', 'Release', *original.monitor.FLAGS,
            '-p:NuGetAudit=false', *command[7:]]
    assert args.count('--tl:off') == args.count('--no-build') == args.count('--no-restore') == 1
    return old_worker(controller, state_path, name, args, cwd, expected, preflight_gib, rss_gib, seconds, allow_children, output)


original.prior = prior; original.monitor.worker = worker


def main():
    assert not BASE.exists(); close_failure()
    original.main()
    value = read(BASE/'inputs.json')
    for p in [FAILED/'failure-closed.json', *PREVIOUS.glob('*')]:
        if p.is_file(): value['files'][p.as_posix()] = pin(p)
    # Freeze the reused implementation in addition to the successor entry points.
    save(BASE/'successor-inputs.json', dict(files=value['files'], corrected_project_invocation=True,
        original_product_source_unchanged=True))


if __name__ == '__main__': main()
