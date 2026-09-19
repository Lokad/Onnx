"""Freeze the closed Windows product and minimal application-reference payload."""
from pathlib import Path
import argparse
import shutil
import subprocess
import tarfile
from remote import h


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact',type=Path,required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    old = root/'artifacts/whisper-recording-v2-20260919'
    assert h.sha(old/'receipt.json') == '623d1ad90efa120d0910f1f9fad92deaf3d4d9ad8410b74e8c10f16742d66275'
    receipt = h.read(old/'receipt.json')
    for name in ('frozen.json','reference-frozen.json','inputs/inputs.json','native-corrected/manifest.json','managed/result.json','audit.json','native-audit.json'):
        assert h.sha(old/name) == receipt['files'][name],name
    assert receipt['all_owned_workers_terminal'] and receipt['payload_files_unchanged']
    frozen = h.read(old/'frozen.json')
    inputs = h.read(old/'inputs/inputs.json')
    native = h.read(old/'native-corrected/manifest.json')
    windows = h.read(old/'managed/result.json')
    base = a.artifact.resolve()
    base.mkdir(exist_ok=True)
    payload = base/'payload'
    payload.mkdir()
    pins,sources,borrowed = {},{},{}
    def add(path,name,digest=None,borrow=None):
        if digest is not None:assert h.sha(path) == digest,str(path)
        pin = dict(bytes=path.stat().st_size,sha256=h.sha(path))
        destination = payload/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():h.verify_file(destination,pin)
        else:shutil.copyfile(path,destination)
        pins[name] = pin
        sources.setdefault(name,[]).append(path.relative_to(root).as_posix())
        if borrow is not None:borrowed[name] = borrow
    for folder,key in [('recording-bin','recording'),('cli-bin','cli')]:
        for filename,digest in frozen['binaries'][key].items():
            if not filename.endswith('.exe'):add(old/folder/filename,'bin/'+filename,digest)
    for name,source in [('reference/windows-receipt.json','receipt.json'),('reference/windows-frozen.json','frozen.json'),
                        ('reference/native.json','native-corrected/manifest.json'),('reference/windows-managed.json','managed/result.json'),
                        ('reference/windows-audit.json','audit.json'),('reference/windows-native-audit.json','native-audit.json'),
                        ('inputs/inputs.json','inputs/inputs.json')]:
        add(old/source,name,receipt['files'].get(source))
    for case in inputs['cases']:
        add(old/'inputs'/case['pcm'],'inputs/'+case['pcm'],case['pcm_sha256'],borrow=case['pcm'])
    first = inputs['cases'][0]
    add(old/'inputs'/first['wave'],'inputs/'+first['wave'],first['wave_sha256'],borrow=first['wave'])
    assets = old/'source/tests/whisper/transcription-assets.json'
    add(assets,'reference/assets.json',frozen['source']['tests/whisper/transcription-assets.json'])
    assert h.read(assets) == native['assets']
    add(old/'source/tests/whisper/recording/Program.cs','reference/Program.cs',frozen['source']['tests/whisper/recording/Program.cs'])
    add(root/'tests/whisper/recording/audit.py','reference/recording_audit.py',receipt['audit_sources']['audit.py'])
    add(old/'source/tests/audio/accuracy/score.py','reference/score.py',frozen['source']['tests/audio/accuracy/score.py'])
    short = root/'artifacts/asr-labeled-20260919/native-whisper/manifest.json'
    add(short,'short/manifest.json',windows['short_manifest_sha256'])
    short_case = h.read(short)['cases'][0]
    add(short.parent/short_case['pcm'],'short/'+short_case['pcm'],short_case['pcm_sha256'])
    support = root/'tests/parakeet/recording-amd/remote.py'
    prior = root/'artifacts/parakeet-recording-amd-20260919/closed.json'
    assert h.sha(prior) == '7d2584e5fe5d78cd8b77ab884139c33fce516ac9b8ef85aa470bb4375a2892e0'
    h.verify_file(support,h.read(prior)['tracked'][support.relative_to(root).as_posix()])
    add(support,'process_support.py')
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.suffix == '.py':add(path,path.name)
    add(root/'.agent/m4-whisper-recording-amd-20260919.md','plan.md')
    bundle = dict(schema=1,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        product_source=frozen['commit'],original_receipt_sha256=h.sha(old/'receipt.json'),
        models='/home/vermorel/Onnx/models/whisper-large-v3-turbo',files=pins,local_sources=sources,borrowed=borrowed)
    h.write_new(payload/'bundle.json',bundle)
    archive = base/'payload.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for path in sorted(payload.rglob('*')):
            if path.is_file() and path.relative_to(payload).as_posix() not in borrowed:
                tar.add(path,arcname=path.relative_to(payload).as_posix(),recursive=False)
    record = dict(bytes=archive.stat().st_size,sha256=h.sha(archive),bundle_sha256=h.sha(payload/'bundle.json'),
        logical_bytes=sum(pin['bytes'] for pin in pins.values()),borrowed_bytes=sum(pins[name]['bytes'] for name in borrowed))
    h.write_new(base/'preparation.json',record)
    print(record)


if __name__ == '__main__':
    main()
