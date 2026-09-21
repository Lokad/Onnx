"""Freeze the tested execution tools separately from the immutable model payload."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from candidate_protocol import LIMITS, pin, read, write
from transport import ROOT, BASE, PREPARED, TOOLS, REMOTE


def selected_native_files(files):
    selected = {}
    packages = ('numpy', 'scipy', 'torch', 'torchaudio', 'onnxruntime', 'pyannote', 'einops', 'pandas', 'sortedcontainers', 'psutil')
    for name, wanted in files.items():
        segment = name.split('/site-packages/', 1)[-1] if '/site-packages/' in name else name.split('/python/', 1)[-1] if '/python/' in name else ''
        package = segment.split('/')[0]
        if any(package == p or package.startswith(p+'.') or package.startswith(p+'-') or package.startswith(p+'_') for p in packages):
            selected[name] = wanted
    return selected


def main():
    verified = read(PREPARED/'preparation-verified.json')
    assert verified['passed'] is True and verified['prepared'] == pin(PREPARED/'prepared.json')
    prepared = read(PREPARED/'prepared.json')
    assert verified['archive'] == prepared['archive'] == pin(PREPARED/'payload.tar.gz')
    failure_path = ROOT/'artifacts/pyannote-amd-execution-20260921/failed-preparation.json'
    assert pin(failure_path)['sha256'] == '67651b666526709a2c90cc4a073e14b6b264ef2f9bf7f65e31b643b0c480309e'
    failure = read(failure_path); assert not failure['passed'] and not failure['inference_executed']
    for name, wanted in failure['files'].items(): assert pin(failure_path.parent/name) == wanted, name
    cancelled_base = ROOT/'artifacts/pyannote-amd-execution-v2-20260921'
    cancellation = read(cancelled_base/'cancelled-for-cli-prerequisite.json')
    assert cancellation['cancelled'] and not cancellation['vm_deployed'] and cancellation['stages_started'] == 0
    assert pin(cancelled_base/'controller/state.json') == cancellation['state']
    state = read(cancelled_base/'controller/state.json'); assert state['stages'] == []
    sys.path.insert(0, str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages')); import psutil
    try: assert psutil.Process(cancellation['identity']['pid']).create_time() != cancellation['identity']['birth']
    except psutil.NoSuchProcess: pass
    cancelled_prepared = read(cancelled_base/'prepared.json')
    assert pin(cancelled_base/'execution.tar.gz') == cancelled_prepared['archive']
    assert pin(cancelled_base/'execution/execution.json') == cancelled_prepared['execution']
    for name, wanted in read(cancelled_base/'execution/execution.json')['files'].items():
        assert pin(cancelled_base/'execution'/name) == wanted, name
    second_base = ROOT/'artifacts/pyannote-amd-execution-v3-20260921'
    second = read(second_base/'cancelled-for-cli-output-drain.json')
    assert second['cancelled'] and not second['vm_deployed'] and second['stages_started'] == 0
    assert pin(second_base/'controller/state.json') == second['state']
    assert read(second_base/'controller/state.json')['stages'] == []
    try: assert psutil.Process(second['identity']['pid']).create_time() != second['identity']['birth']
    except psutil.NoSuchProcess: pass
    assert pin(second_base/'execution.tar.gz') == read(second_base/'prepared.json')['archive']
    BASE.mkdir(); target = BASE/'execution'; target.mkdir()
    local_tools = {}
    for path in sorted(TOOLS.glob('*.py')):
        ast.parse(path.read_text(encoding='utf8'), filename=str(path))
        local_tools[path.relative_to(ROOT).as_posix()] = pin(path)
        shutil.copy2(path, target/path.name)
    original = ROOT/'tests/parakeet/transcribe/audit.py'
    local_tools[original.relative_to(ROOT).as_posix()] = pin(original)
    shutil.copy2(original, target/'parakeet_audit.py')
    with (BASE/'selftest.log').open('x') as log:
        code = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover', '-s', str(TOOLS), '-p', 'test_*.py', '-v'],
                              cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120).returncode
    assert code == 0, 'Execution selftests failed; retain this attempt'
    old = ROOT/'artifacts/audio-amd-two-family-20260920/collected/frozen.json'
    closed = read(ROOT/'artifacts/audio-amd-two-family-20260920/closed.json')
    key = next(k for k in closed['files'] if Path(k).resolve() == old.resolve())
    assert pin(old) == closed['files'][key]
    retained = read(old); external = selected_native_files(retained['external'])
    assert len(external) == 16117
    execution = dict(schema=1, source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                     payload=prepared['payload'], payload_archive=prepared['archive'], remote=REMOTE, limits=LIMITS,
                     local_tools=local_tools, files={p.name: pin(p) for p in target.iterdir() if p.is_file()}, external=external,
                     native_predecessor=pin(old), failed_predecessor=pin(failure_path), selftest=pin(BASE/'selftest.log'),
                     cancelled_predecessor=pin(cancelled_base/'cancelled-for-cli-prerequisite.json'),
                     second_cancelled_predecessor=pin(second_base/'cancelled-for-cli-output-drain.json'),
                     scope='Tested offline execution tools; AMD inference has not run. Existing e5 owns the VM until terminal.')
    write(target/'execution.json', execution)
    with tarfile.open(BASE/'execution.tar.gz', 'w:gz') as archive:
        for path in sorted(target.iterdir()): archive.add(path, arcname=path.name, recursive=False)
    receipt = dict(passed=True, execution=pin(target/'execution.json'), archive=pin(BASE/'execution.tar.gz'),
                   payload_archive=prepared['archive'], local_preparation=pin(PREPARED/'preparation-verified.json'),
                   files=len(execution['files']), supplemental_native_files=len(external),
                   supplemental_native_bytes=sum(v['bytes'] for v in external.values()), selftest=execution['selftest'],
                   amd_deployed=False, amd_qualified=False, amd_timing=False)
    write(BASE/'prepared.json', receipt); print(json.dumps(receipt))


if __name__ == '__main__':
    main()
