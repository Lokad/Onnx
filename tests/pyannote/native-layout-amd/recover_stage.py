"""Complete the already-extracted stage after invalidating Python's path cache.

The original stage imported hashlib with tools/ in sys.path before extraction.
Python cached that then-missing directory. Preserve that failed stage; do not
change its payload, recreate its directory, or launch a second diagnostic.
"""
import json
from run import BASE, PRELUDE, prepared, previous_closed, ssh
from protocol import pin, save


def main():
    spec = prepared(); previous_closed()
    assert not (BASE/'staged.json').exists() and not (BASE/'deployment.json').exists()
    failure = BASE/'stage-failure.json'; assert not failure.exists()
    save(failure,dict(preserved=True,stage_exit_code=1,
        error="ModuleNotFoundError: No module named 'protocol'",
        phase='After verified archive extraction, before remote verification or deployment',
        archive=spec['archive'],payload=spec['payload'],original_tools=spec['files'],
        recovery_tool=pin(__file__)))
    receipt = json.loads(ssh(PRELUDE+f'''
import importlib,hashlib
assert base.is_dir() and not (base/'staged.json').exists()
assert not (base/'deployment.json').exists() and not (base/'identity.json').exists()
assert (base/'tools/protocol.py').is_file()
archive=base/'transfer.tar.gz'
with archive.open('rb') as f: actual=dict(bytes=archive.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
assert actual=={spec['archive']!r}
importlib.invalidate_caches()
from protocol import pin,save,verify
import remote
remote.idle()
assert pin(base/'payload.json')=={spec['payload']!r}
value=verify(base);assert not remote.live(value['previous_owner'])
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']),
 recovery='Invalidate importer cache for the directory created by the preceding extraction; same payload and no prior deployment')
save(base/'staged.json',receipt);print(json.dumps(receipt))
''',300))
    save(BASE/'staged.json',receipt)
    save(BASE/'stage-recovery.json',dict(passed=True,failure=pin(failure),tool=pin(__file__),receipt=pin(BASE/'staged.json'),no_payload_change=True))
    print(json.dumps(receipt))


if __name__ == '__main__': main()
