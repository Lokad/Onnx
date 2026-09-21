"""Prepare a full e5 payload locally; no VM access or inference."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

from contract import pin, read, write, CORE, LIMITS, CRITERIA
from design import PROTOCOL, assignment_schedule
from remote import ROOT, BASE, WHISPER


def main():
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Commit tools first'
    assert not (BASE/'payload').exists() and not (BASE/'prepared.json').exists()
    verified = read(BASE/'runner-verification.json'); assert verified['passed'] is True
    assert pin(BASE/'runner-verification.json') == dict(bytes=30208, sha256='548abbea45234903e21d7ca69e20c079f6d9012cc6fc92ebd8c31922f5fcc5cb')
    for name, wanted in verified['files'].items():
        assert pin(BASE/name) == wanted, name
    for name, key in [('contract.py', 'contract'), ('run.py', 'runner')]:
        assert pin(Path(__file__).with_name(name)) == verified['source_changes'][key], name
    assignments = read(BASE/'assignments-frozen.json')
    assert pin(BASE/'assignments-frozen.json') == dict(bytes=889069, sha256='2ac2bd42acfc811b39f2df099c7e93f31789bea18236e0e507d74e60ba5b16f5')
    for phase in ('aa', 'compare'):
        assert assignments['schedules'][phase] == assignment_schedule(assignments['draws'][phase], phase)
    original = ROOT/'artifacts/e5-process-uncertainty-20260921'
    qualification = read(original/'local-verification.json'); assert qualification['passed'] is True
    for name, wanted in qualification['pins'].items():
        assert pin(original/name) == wanted, name
    assert pin(original/'bin/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(WHISPER/'frozen.json') == dict(bytes=4401899, sha256='0fafe2763d111a63a4bc0f8f5aad45eb3a911e2ade23d9dacfde0eeab6a7e74d')
    inherited = read(WHISPER/'frozen.json'); runtime = inherited['managed_runtime']
    assert runtime['host'] == '/home/vermorel/.dotnet/dotnet'
    native = ROOT/'artifacts/e5-public-ort-20260919/bin/libonnxruntime.so'
    assert pin(native)['sha256'] == '13ab8084954fa4a47c777880180b90810d6020f021441395712b48a75b74c68b'
    # NumPy comes from the interpreter's user site, not one of the three
    # additional audio package paths. Bind its already recorded files too.
    numpy_roots = {str(Path(path).parent.parent).replace('\\', '/') for path in inherited['external']
                   if path.endswith('/site-packages/numpy/__init__.py')}
    assert len(numpy_roots) == 1
    python_files = {}
    for prefix in set(inherited['python_paths']) | numpy_roots:
        for path, identity in inherited['external'].items():
            if path.startswith(prefix+'/'):
                package = path[len(prefix)+1:].split('/')[0]
                if package in ('numpy', 'numpy.libs', 'psutil') or package.startswith(('numpy-', 'psutil-')):
                    python_files[path] = identity
    assert python_files and any('/numpy/' in p for p in python_files) and any('/psutil/' in p for p in python_files)
    payload = BASE/'payload'; payload.mkdir()
    shutil.copytree(original/'bin', payload/'bin'); shutil.copytree(original/'inputs', payload/'inputs')
    shutil.copyfile(BASE/'assignments-frozen.json', payload/'assignments-frozen.json')
    for directory, names in [
        ('randomized-processes', ['design.py', 'contract.py', 'run.py']),
        ('process-uncertainty', ['estimator.py', 'protocol.py', 'worker_audit.py'])]:
        target = payload/'tests/e5'/directory; target.mkdir(parents=True)
        for name in names:
            shutil.copyfile(ROOT/'tests/e5'/directory/name, target/name)
    (payload/'eng').mkdir(); shutil.copyfile(ROOT/'eng/campaign_processes.py', payload/'eng/campaign_processes.py')
    shutil.copyfile(ROOT/'.agent/m1-randomized-processes-20260921.md', payload/'prospective-plan.md')
    model = ROOT/'models/multilingual-e5-small/model.onnx'
    meta = dict(protocol=PROTOCOL, worker_protocol='e5-fresh-process-uncertainty-v1', mode='full',
                source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                product_source='0f86c5d', core_sha256=CORE, limits=LIMITS, criteria=CRITERIA,
                diagnostic_limits=dict(maximum_observed_variance_share=.2),
                schedules=assignments['schedules'], model=dict(path='/home/vermorel/Onnx/models/multilingual-e5-small/model.onnx', **pin(model)),
                native=dict(path='/home/vermorel/Onnx/artifacts/e5-public-ort-20260919/worker/libonnxruntime.so', **pin(native)),
                dotnet=dict(path=runtime['host'], **runtime['files'][runtime['host']]),
                runtime_files=[dict(path=name, **identity) for name, identity in sorted(runtime['files'].items())],
                python_paths=inherited['python_paths'], python_files=python_files, interpreter=inherited['interpreter'],
                predecessor_scope='audio-whisper-amd-20260921', predecessor_births=None,
                qualification=dict(producer=pin(original/'local-verification.json'), runner=pin(BASE/'runner-verification.json'),
                                   simulation=pin(BASE/'simulation-verification.json'), runtime_source=pin(WHISPER/'frozen.json')),
                files={p.relative_to(payload).as_posix(): pin(p) for p in sorted(payload.rglob('*')) if p.is_file()})
    write(BASE/'full-template.json', meta)
    with tarfile.open(BASE/'payload.tar.gz', 'x:gz') as archive:
        for name in meta['files']:
            archive.add(payload/name, arcname=name, recursive=False)
    # Actual predecessor births and Python executable path are bound on the idle
    # VM before frozen.json is created. No measurement fields may change there.
    write(BASE/'prepared.json', dict(passed=True, template=pin(BASE/'full-template.json'), archive=pin(BASE/'payload.tar.gz'),
                                   files=len(meta['files']), runtime_files=len(meta['runtime_files']), python_files=len(python_files),
                                   worker_assembly=pin(payload/'bin/ProcessUncertainty.dll'), core=pin(payload/'bin/Lokad.Onnx.dll')))
    print(json.dumps(read(BASE/'prepared.json')))


if __name__ == '__main__':
    main()
