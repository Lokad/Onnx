"""Freeze existing qualified binaries for six AMD numerical workers, without rebuilding."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import JOBS, LIMITS, pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-filter-block-reuse-amd-20260922'
LOCAL = ROOT/'artifacts/pyannote-filter-block-reuse-20260922'
APP = ROOT/'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922'
PREVIOUS = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v3-20260922'
FIXTURES = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922/output'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
CORE = '64f42e978ccea3d38e38bd30eb372a384653628bc1b3a7b92f2d2224f16f3c8e'
spec = importlib.util.spec_from_file_location('filter_block_amd_local_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)


def previous_closed():
    assert pin(LOCAL/'closed.json')['sha256'] == 'c937aeb7aa6061623cc1c5bab77a09c920374d6a8ff84c68c6f555dd821051a4'
    proof = read(LOCAL/'closed.json'); assert proof['passed'] and not proof['avx512_qualified']
    for name, wanted in proof['files'].items(): assert pin(LOCAL/name) == wanted, name
    for identity in proof['identities']: monitor.terminal(identity)
    for filename in ['inputs.json', 'consumer-inputs.json']:
        monitor.verify(read(LOCAL/filename)['files'])
    assert pin(LOCAL/'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(APP/'closed.json')['sha256'] == '5c238cd33845eb58fc00332361a967185ae82a9fcb70854530e07fad58f064d0'
    assert read(APP/'closed.json')['passed'] and read(APP/'analysis.json')['performance']['admitted']
    receipt = read(APP/'collected/collection.json'); assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    owner = {key: receipt['identities'][0][key] for key in ['pid', 'birth']}
    assert owner == dict(pid=707355, birth=1790085675.48)
    return owner


def prepare():
    assert not BASE.exists(); owner = previous_closed()
    old_proof = read(PREVIOUS/'closed.json')
    assert pin(PREVIOUS/'closed.json')['sha256'] == '88ec9b71d6e7807440ccd65d6148ead8c585a0633ce755898e174b8501b77197'
    assert old_proof['passed']
    old = read(PREVIOUS/'payload/payload.json')
    assert pin(PREVIOUS/'payload/payload.json') == old_proof['files']['payload/payload.json']
    BASE.mkdir(); payload = BASE/'payload'; payload.mkdir(); (payload/'tools').mkdir(); (payload/'fixtures').mkdir()
    consumers = {}
    for mode in ['raw', 'wide', 'layers']:
        built = LOCAL/'consumers'/mode/'bin/Release/net10.0'
        shutil.copytree(built, payload/'runtime'/mode)
        assert pin(built/'Lokad.Onnx.dll')['sha256'] == CORE
        assembly = 'LayerGraphs.dll' if mode == 'layers' else 'Lokad.Onnx.Backend.Tests.dll'
        consumers[mode] = pin(built/assembly)
        expected = read(LOCAL/'output'/(mode+'-256.json'))
        assert expected['executable'] == consumers[mode]['sha256'] and expected['core'] == CORE and expected['passed']
        shutil.copy2(LOCAL/'output'/(mode+'-256.json'), payload/('windows-'+mode+'.json'))
    for name in ['remote.py', 'protocol.py', 'checks.py']: shutil.copy2(TOOLS/name, payload/'tools'/name)
    shutil.copy2(FIXTURES/'result.json', payload/'fixtures/result.json')
    shutil.copy2(ROOT/'.agent/m18-pyannote-filter-block-reuse-20260922.md', payload/'prospective-plan.md')
    external = dict(old['external'])
    for p in FIXTURES.iterdir():
        if p.is_file(): assert external[old['fixture_directory']+'/'+p.name] == pin(p), p.name
    manifest = dict(passed=True, limits=LIMITS, jobs=JOBS, previous_owner=owner, boot_time=1789634288.0,
        interpreter=old['interpreter'], external=external, fixture_directory=old['fixture_directory'],
        core=pin(LOCAL/'runtime/Lokad.Onnx.dll'), consumers=consumers,
        files={p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()},
        scope='Original and wide raw graph families plus all captured layers, actual AMD AVX2/AVX512; no timing or codegen claim')
    save(payload/'payload.json', manifest)
    files = {p.relative_to(ROOT).as_posix(): pin(p) for p in [*TOOLS.iterdir(), MONITOR, LOCAL/'closed.json',
        APP/'closed.json', APP/'analysis.json', APP/'collected/collection.json', PREVIOUS/'closed.json', PREVIOUS/'payload/payload.json'] if p.is_file()}
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(payload.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(payload).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=files, payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz'), jobs=JOBS, core=manifest['core'], consumers=consumers)))


if __name__ == '__main__': prepare()
