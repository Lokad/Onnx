"""Freeze exact previously qualified portable binaries, inputs and native reference."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tarfile
from remote import read, sha, verify_file, write_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    old = root/'artifacts/parakeet-recording-20260919'
    receipt = read(old/'receipt.json')
    assert receipt['closed'] and sha(old/'receipt.json') == '2f23617863b6718677578b27357528dc1f8ece5e84cd0112a8cd7293839a65e6'
    base = a.artifact.resolve()
    base.mkdir(exist_ok=True)
    payload = base/'payload'
    payload.mkdir()
    pins, sources, recipes = {}, {}, {}

    def add(path, name, old_name=None):
        if old_name is not None:
            verify_file(path, receipt['files'][old_name])
        destination = payload/name
        pin = dict(bytes=path.stat().st_size, sha256=sha(path))
        if destination.exists():
            verify_file(destination, pin)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        pins[name] = pin
        sources.setdefault(name, []).append(old_name if old_name is not None else str(path.relative_to(root)))

    for folder in ('bin', 'cli-bin'):
        for path in sorted((old/folder).iterdir()):
            if path.is_file() and path.suffix != '.exe':
                add(path, 'bin/'+path.name, folder+'/'+path.name)
    inputs = read(old/'inputs/inputs.json')
    seen = {}
    for case in inputs['cases']:
        name = 'inputs/'+case['pcm']
        path = old/name
        verify_file(path, receipt['files'][name])
        assert sha(path) == case['pcm_sha256']
        pin = dict(bytes=path.stat().st_size, sha256=sha(path))
        if case['name'] == 'maximum-silence':
            raw = path.read_bytes()
            assert raw[:6] == b'\x93NUMPY' and raw[6:8] == b'\x01\x00'
            header_end = 10 + int.from_bytes(raw[8:10], 'little')
            assert raw[header_end:] == bytes(9600000*4)
            recipes[name] = dict(kind='sparse-zero', header_hex=raw[:header_end].hex())
        elif pin['sha256'] in seen:
            recipes[name] = dict(kind='hardlink', source=seen[pin['sha256']])
        else:
            add(path, name, name)
            seen[pin['sha256']] = name
        pins[name] = pin
        sources[name] = [name]
    for name in ('inputs/inputs.json', 'inputs/connected.wav'):
        add(old/name, name, name)
    for old_name, name in [
        ('native/result.json', 'reference/native.json'),
        ('managed/result.json', 'reference/windows-managed.json'),
        ('frozen.json', 'reference/windows-frozen.json'),
        ('closure-source/tests/parakeet/recording/audit.py', 'reference/recording-audit.py'),
        ('source/tests/parakeet/transcribe/assets.json', 'reference/assets.json'),
        ('source/tests/parakeet/recording/Program.cs', 'reference/Program.cs'),
    ]:
        add(old/old_name, name, old_name)
    add(old/'receipt.json', 'reference/windows-receipt.json')
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.suffix == '.py':
            add(path, path.name)
    add(root/'.agent/m4-parakeet-recording-amd-20260919.md', 'plan.md')
    bundle = dict(schema=1, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        implementation_commit=receipt['implementation_commit'], original_receipt_sha256=sha(old/'receipt.json'),
        models='/home/vermorel/Onnx/models/parakeet-tdt-0.6b-v3', files=pins, recipes=recipes, original_paths=sources)
    write_new(payload/'bundle.json', bundle)
    archive = base/'payload.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted(payload.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(payload).as_posix(), recursive=False)
    record = dict(bytes=archive.stat().st_size, sha256=sha(archive), bundle_sha256=sha(payload/'bundle.json'),
        logical_bytes=sum(pin['bytes'] for pin in pins.values()), sparse_or_link_bytes=sum(pins[n]['bytes'] for n in recipes))
    write_new(base/'preparation.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
