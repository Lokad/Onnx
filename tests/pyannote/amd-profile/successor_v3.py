"""Preserve the strict instruction gate by placing new imports after existing methods."""
import sys
import common

FAILED = common.ROOT / 'artifacts/pyannote-amd-profile-v2-20260922'
common.BASE = common.ROOT / 'artifacts/pyannote-amd-profile-v3-20260922'
common.REMOTE = '/dev/shm/lokad-pyannote-amd-profile-v3-20260922'
common.monitor.BASE = common.BASE


def main():
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare','stage','launch','observe','collect','export','audit']
    failure = common.read(FAILED / 'preparation-failure.json'); assert failure['preserved']; common.verify(failure['files'])
    for identity in failure['identities']: common.terminal(identity)
    action = sys.argv[1]
    path = common.TOOLS / ('prepare.py' if action == 'prepare' else 'transport.py' if action in ['stage','launch','observe','collect'] else action+'.py')
    source = path.read_text(encoding='utf8')
    if action == 'prepare':
        imports = '    [DllImport("kernel32.dll", EntryPoint = "GetCurrentThreadId")] static extern uint WindowsThreadId();\n    [DllImport("libc", EntryPoint = "gettid")] static extern uint LinuxThreadId();\n'
        changes = [
            (imports, ''),
            ("after = before.replace(old, new); p.write_text(after, encoding='utf8')",
             "after = before.replace(old, new); marker = '\\n}\\n\\nstatic class SampledRequests'; assert after.count(marker) == 1; after = after.replace(marker, '\\n' + " + repr(imports) + " + marker); p.write_text(after, encoding='utf8')"),
            ("pin(PAYLOAD / name) == wanted == archive['files'][name]", "pin(PAYLOAD / 'assets' / name) == wanted == archive['files']['assets/' + name]"),
            ("copy(PAYLOAD / name, payload / name)", "copy(PAYLOAD / 'assets' / name, payload / name)"),
            ("files[rel(original_pair)] = pin(original_pair)", "files[rel(original_pair)] = pin(original_pair)\n    files[rel(ROOT / 'artifacts/pyannote-amd-profile-v2-20260922/preparation-failure.json')] = pin(ROOT / 'artifacts/pyannote-amd-profile-v2-20260922/preparation-failure.json')"),
        ]
        for old, new in changes:
            assert source.count(old) == 1, old; source = source.replace(old, new)
    namespace = dict(__name__='amd_profile_v3_'+action, __file__=str(path))
    exec(compile(source, str(path), 'exec'), namespace)
    namespace[action if path.name == 'transport.py' else 'main']()


if __name__ == '__main__': main()
