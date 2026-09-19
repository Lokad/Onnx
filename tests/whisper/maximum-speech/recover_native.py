"""Close a successful native worker after the preserved immediate-exit assertion failed.

This never changes the original supervisor evidence or reruns inference. It only
accepts the specific post-wait assertion after all born identities are absent,
every payload/reference array remains exact, and the independent oracle passes.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import psutil
from tokenizers import Tokenizer
from prepare import sha, read, write_new, pin
from audit import process, validator_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    args = parser.parse_args()
    base = args.artifact.resolve()
    assert not (base/'native-terminal-recovery.json').exists()
    identity = read(base/'native-process.json')
    frozen = read(base/'frozen.json')
    for name,expected in frozen['files'].items():
        assert pin(base/name) == expected,name
    births = {(int(pid),birth) for pid,birth in identity['members'].items()}|{(identity['supervisor'],identity['supervisor_create_time'])}
    for pid,birth in births:
        try:
            assert psutil.Process(pid).create_time() != birth, 'An owned identity remains'
        except psutil.NoSuchProcess:
            pass
    native,independent,inputs = read(base/'native/manifest.json'),read(base/'native-audit.json'),read(base/'inputs/inputs.json')
    assert independent['passed'] and independent['native_sha256'] == sha(base/'native/manifest.json')
    assert independent['inputs_sha256'] == native['inputs_sha256'] == frozen['inputs_sha256'] == sha(base/'inputs/inputs.json')
    assert independent['auditor_sha256'] == sha(base/'native-source/recording/audit_native.py')
    assert native['generator_sha256'] == sha(base/'native-source/recording/generate_reference.py')
    assert native['timestamp_helper_sha256'] == independent['timestamp_helper_sha256'] == sha(base/'native-source/recording/generate_rules.py')
    validator = validator_at(base/'reference')
    tokenizer = Tokenizer.from_file(str(Path(frozen['models'])/'tokenizer.json'))
    assert len(native['cases']) == len(inputs['cases']) == 1
    result = native['cases'][0]['result']
    validator.validate_recording(result,inputs['cases'][0],tokenizer)
    assert result['stop_reason'] == 'Completed' and result['processed_seconds'] == 600
    assert independent['steps'] == sum(len(w['decoding']['token_ids']) for w in result['windows'])
    expected = set()
    for i,window in enumerate(result['windows']):
        prefix = f'maximum-speech-w{i:03d}'
        expected.update([prefix+'-features.npy',prefix+'-hidden.npy'])
        expected.update(prefix+f'-{step:03d}-logits.npy' for step in range(len(window['decoding']['token_ids'])))
    assert set(native['files']) == expected == {p.name for p in (base/'native').glob('*.npy')}
    for name,item in native['files'].items():
        assert pin(base/'native'/name) == {key:item[key] for key in ('bytes','sha256')}
        array = np.load(base/'native'/name,allow_pickle=False,mmap_mode='r')
        assert array.dtype == np.float32 and list(array.shape) == item['shape'] and np.isfinite(array).all()
    samples = [json.loads(line) for line in (base/'native-samples.jsonl').read_text(encoding='utf-8').splitlines()]
    recovery = dict(schema=1,passed=True,original_identity=identity,terminal_check_time=time.time(),
        terminal_processes=[dict(pid=pid,create_time=birth) for pid,birth in sorted(births)],
        frozen_sha256=sha(base/'frozen.json'),native_process_sha256=sha(base/'native-process.json'),
        native_sha256=sha(base/'native/manifest.json'),native_audit_sha256=sha(base/'native-audit.json'),
        verifier_sha256=sha(Path(__file__)),native_arrays=len(expected),
        scope='Worker exited0; original immediate-exit assertion failure preserved; all identities subsequently absent and complete reference reverified. No inference rerun.')
    recovery['resources'] = process(identity,samples,'native',sha(base/'frozen.json'),recovery)
    write_new(base/'native-terminal-recovery.json',recovery)
    print(json.dumps({k:recovery[k] for k in ('passed','terminal_processes','native_arrays','resources')},indent=2))


if __name__ == '__main__':
    main()
