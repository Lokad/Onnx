"""Finish the unstarted decoder preparation after the extraction import-cache failure."""
import ast
import base64
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests/parakeet/decoder-projection-observation'))
import run
from protocol import pin, read, save

OUT = ROOT/'artifacts/parakeet-decoder-projection-stage-recovery-20260927'


def main():
    assert not OUT.exists(), 'Preserve the one-time recovery attempt'
    spec = run.prepared()
    assert (run.BASE/'stage-started.json').is_file()
    assert not any((run.BASE/name).exists() for name in
                   ['extracted.json', 'payload.json', 'staged.json', 'deployment.json', 'closed.json'])
    expressions = [node.args[0] for node in ast.walk(ast.parse(inspect.getsource(run.stage)))
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ssh']
    assert len(expressions) == 3
    # Reuse exactly the original remaining preparation, after a fresh interpreter
    # has verified the already extracted files. Never repeat extraction or inference.
    remaining = eval(compile(ast.Expression(expressions[2]), '<remaining-stage>', 'eval'),
                     dict(vars(run), spec=spec))
    compile(remaining, '<remaining-stage>', 'exec')
    assert "tools/remote_prepare.py" in remaining and 'tar.extractall' not in remaining
    OUT.mkdir()
    save(OUT/'intent.json', dict(
        failure="ModuleNotFoundError: No module named 'protocol' after tar extraction",
        cause='The newly created tools directory retained a negative Python importer-cache entry; the inherited extraction invalidates caches, this adapter omitted that step.',
        prepared=pin(run.BASE/'prepared.json'), stage_started=pin(run.BASE/'stage-started.json'),
        original_transport=pin(ROOT/'tests/parakeet/decoder-projection-observation/run.py'),
        recovery_tool=pin(Path(__file__)),
        scope='Verify the complete extracted input and absence of any campaign, then execute only the original unstarted preparation in a fresh process.'))
    (OUT/'remaining-stage.py').write_text(remaining, encoding='utf8')
    preflight = json.loads(run.ssh(run.PRELUDE+f'''
import importlib
importlib.invalidate_caches()
from protocol import read,pin,LIMITS
import remote
remote.idle()
assert psutil.boot_time()==1789634288.0
assert psutil.virtual_memory().available>=LIMITS['preflight_available']
assert psutil.disk_usage('/dev/shm').free>=LIMITS['preflight_tmpfs']
assert not any((base/name).exists() for name in ['payload.json','identity.json','deployment.json','staged.json','stage-prepare.stdout','stage-prepare.stderr'])
assert pin(base/'stage.json')=={spec['stage']!r}
assert pin(base/'transfer.tar.gz')=={spec['archive']!r}
stage=read(base/'stage.json')
assert {{p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}}==set(stage['files'])|{{'stage.json','transfer.tar.gz'}}
for name,wanted in stage['files'].items():assert pin(base/name)==wanted,name
assert not any((base/name).exists() for name in stage['links'])
root=Path('/dev/shm/lokad-parakeet-rational-sigmoid-root-20260927')
prior=read(root/'collection.json')
assert prior['terminal'] and prior['code']==0 and not any(remote.live(i) for i in prior['identities'])
result=dict(passed=True,extracted_files=len(stage['files']),no_campaign_started=True,
    original_stage=pin(base/'stage.json'),archive=pin(base/'transfer.tar.gz'),
    available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
# The original stage retired this transfer after verification. Its exact local
# archive and identity remain preserved in the prepared campaign.
(base/'transfer.tar.gz').unlink()
print(json.dumps(result))
''', 60))
    save(OUT/'verified-extraction.json', preflight)
    save(run.BASE/'extracted.json', dict(passed=True, archive_retired=True,
        recovery_intent=dict(path=(OUT/'intent.json').relative_to(ROOT).as_posix(), identity=pin(OUT/'intent.json'))))
    raw = run.ssh(remaining, 240)
    (OUT/'remaining-stage.stdout').write_text(raw, encoding='utf8')
    staged = json.loads(raw)
    encoded = staged.pop('payload_base64')
    (run.BASE/'payload.json').write_bytes(base64.b64decode(encoded))
    assert pin(run.BASE/'payload.json') == staged['payload'] and staged['passed']
    save(run.BASE/'staged.json', staged)
    save(OUT/'closed.json', dict(passed=True, campaign_started=False,
        files={p.name: pin(p) for p in OUT.iterdir() if p.is_file()},
        extracted=pin(run.BASE/'extracted.json'), staged=pin(run.BASE/'staged.json'),
        payload=pin(run.BASE/'payload.json')))
    print(json.dumps(dict(passed=True, recovery=pin(OUT/'closed.json'), staged=staged)))


if __name__ == '__main__': main()
