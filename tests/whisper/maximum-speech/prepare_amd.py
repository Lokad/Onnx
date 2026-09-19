"""Prepare the same runner for AMD after independently auditing the native oracle."""
from pathlib import Path
import argparse
import subprocess
import tarfile
from prepare import sha, read, write_new, pin, copy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = args.windows.resolve()
    frozen,native,audit = read(source/'frozen.json'),read(source/'native/manifest.json'),read(source/'native-audit.json')
    assert audit['passed'] and audit['native_sha256'] == sha(source/'native/manifest.json')
    assert audit['inputs_sha256'] == native['inputs_sha256'] == frozen['inputs_sha256'] == sha(source/'inputs/inputs.json')
    for name,expected in frozen['files'].items():
        assert pin(source/name) == expected,name
    result = native['cases'][0]['result']
    assert result['stop_reason'] == 'Completed' and result['processed_seconds'] == 600 and len(result['windows']) >= 20
    native_process = read(source/'native-process.json')
    if not native_process['complete']:
        recovery = read(source/'native-terminal-recovery.json')
        assert recovery['passed'] and recovery['original_identity'] == native_process
        assert recovery['native_process_sha256'] == sha(source/'native-process.json')
        assert recovery['native_sha256'] == sha(source/'native/manifest.json')
        assert recovery['native_audit_sha256'] == sha(source/'native-audit.json')
    assert native_process['code'] == 0
    import psutil
    for pid,birth in list(native_process['members'].items())+[(native_process['supervisor'],native_process['supervisor_create_time'])]:
        try:
            assert psutil.Process(int(pid)).create_time() != birth,'Native process remains'
        except psutil.NoSuchProcess:
            pass
    base = args.artifact.resolve()
    base.mkdir(parents=True,exist_ok=False)
    payload = base/'payload'
    payload.mkdir()
    files,borrowed = {},{}
    whisper = '/home/vermorel/Onnx/artifacts/whisper-recording-amd-v2-20260919'
    parakeet = '/home/vermorel/Onnx/artifacts/parakeet-recording-amd-20260919'
    def add(path,name,borrow=None):
        copy(path,payload/name)
        files[name] = pin(path)
        if borrow is not None:
            borrowed[name] = borrow
    for path in sorted((source/'bin').iterdir()):
        if path.suffix not in ('.dll','.json'):
            continue
        name = 'bin/'+path.name
        borrow = whisper+'/'+name if path.name != 'RecordingReplay.dll' else None
        add(path,name,borrow)
    for filename in ('inputs.json','maximum-speech.npy'):
        name = 'inputs/'+filename
        add(source/name,name,parakeet+'/'+name if filename.endswith('.npy') else None)
    for path in (source/'short').iterdir():
        add(path,'short/'+path.name,whisper+'/short/'+path.name)
    for name,filename in [('reference/native.json','native/manifest.json'),('reference/native-audit.json','native-audit.json'),
                          ('reference/windows-frozen.json','frozen.json'),('reference/assets.json','native-source/transcription-assets.json'),
                          ('reference/recording_audit.py','reference/recording_audit.py'),('reference/score.py','reference/score.py')]:
        add(source/filename,name)
    if (source/'native-terminal-recovery.json').exists():
        add(source/'native-terminal-recovery.json','reference/native-terminal-recovery.json')
    support = root/'tests/parakeet/recording-amd/remote.py'
    old = root/'artifacts/parakeet-recording-amd-20260919/closed.json'
    assert sha(old) == '7d2584e5fe5d78cd8b77ab884139c33fce516ac9b8ef85aa470bb4375a2892e0'
    assert pin(support) == read(old)['tracked'][support.relative_to(root).as_posix()]
    add(support,'process_support.py')
    for path in Path(__file__).parent.glob('*.py'):
        add(path,path.name)
    add(root/'.agent/m4-whisper-maximum-speech-20260919.md','plan.md')
    bundle = dict(schema=1,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        product_source=frozen['product_source'],windows_frozen_sha256=sha(source/'frozen.json'),
        models='/home/vermorel/Onnx/models/whisper-large-v3-turbo',files=files,borrowed=borrowed)
    write_new(payload/'bundle.json',bundle)
    archive = base/'payload.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for path in sorted(payload.rglob('*')):
            if path.is_file() and path.relative_to(payload).as_posix() not in borrowed:
                tar.add(path,arcname=path.relative_to(payload).as_posix(),recursive=False)
    preparation = dict(**pin(archive),bundle_sha256=sha(payload/'bundle.json'),
        logical_bytes=sum(p['bytes'] for p in files.values()),borrowed_bytes=sum(files[name]['bytes'] for name in borrowed))
    write_new(base/'preparation.json',preparation)
    print(preparation)


if __name__ == '__main__':
    main()
