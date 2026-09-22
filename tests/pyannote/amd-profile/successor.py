"""Retain the pre-build path failure; use a fresh destination and pinned asset root."""
import sys
import common

FAILED = common.BASE
common.BASE = common.ROOT / 'artifacts/pyannote-amd-profile-v2-20260922'
common.REMOTE = '/dev/shm/lokad-pyannote-amd-profile-v2-20260922'
common.monitor.BASE = common.BASE


def main():
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare','stage','launch','observe','collect','export','audit']
    failure = common.read(FAILED / 'preparation-failure.json')
    assert failure['preserved']
    common.verify(failure['files'])
    action = sys.argv[1]
    script = 'prepare.py' if action == 'prepare' else 'transport.py' if action in ['stage','launch','observe','collect'] else action+'.py'
    path = common.TOOLS / script
    source = path.read_text(encoding='utf8')
    if action == 'prepare':
        changes = [
            ("pin(PAYLOAD / name) == wanted == archive['files'][name]", "pin(PAYLOAD / 'assets' / name) == wanted == archive['files']['assets/' + name]"),
            ("copy(PAYLOAD / name, payload / name)", "copy(PAYLOAD / 'assets' / name, payload / name)"),
            ("files[rel(original_pair)] = pin(original_pair)", "files[rel(original_pair)] = pin(original_pair)\n    files[rel(ROOT / 'artifacts/pyannote-amd-profile-20260922/preparation-failure.json')] = pin(ROOT / 'artifacts/pyannote-amd-profile-20260922/preparation-failure.json')"),
        ]
        for old, new in changes:
            assert source.count(old) == 1, old
            source = source.replace(old, new)
    namespace = dict(__name__='amd_profile_successor_'+action, __file__=str(path))
    exec(compile(source, str(path), 'exec'), namespace)
    namespace[action if script == 'transport.py' else 'main']()


if __name__ == '__main__': main()
