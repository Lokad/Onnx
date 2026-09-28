"""Select supported local closure maps before the unchanged verified consolidation."""
import json
from pathlib import Path
import runpy

initial = Path(__file__).with_name('dedupe-closed-parakeet-september27.py')
worker = runpy.run_path(str(initial), run_name='retained_initial_selector')['worker']
root, artifacts = worker['ROOT'], worker['ARTIFACTS']
assert not worker['OUT'].exists(), 'Initial preflight must have made no filesystem replacements'
failure = artifacts/'repository-retention-20260923/closed-september27-preflight-failed-20260928.json'
assert worker['pin'](initial) == json.loads(failure.read_text())['script']
worker['OUT'] = artifacts/'repository-retention-20260923/closed-september27-filtered-dedup-20260928'
selected, excluded = [], []
for pattern in worker['PATTERNS']:
    for folder in sorted(artifacts.glob(pattern)):
        closure = folder/'closed.json'
        if not closure.is_file(): continue
        value = json.loads(closure.read_text())
        if not (value.get('passed') or value.get('completed')): continue
        files = value.get('files')
        if not isinstance(files, dict):
            excluded.append(dict(folder=folder.name, reason='Closure does not contain a file identity map'))
            continue
        unsupported = []
        for relative, wanted in files.items():
            if not isinstance(wanted, dict) or wanted.get('bytes', 0) < 4096: continue
            path = folder/relative
            if path.suffix not in worker['SUFFIXES']: continue
            if not path.exists() or path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                unsupported.append(relative)
        if unsupported:
            excluded.append(dict(folder=folder.name, reason='Closure uses another path schema or aliases', paths=unsupported))
        else: selected.append(folder.name)
assert selected and len(selected) == len(set(selected))
worker['save'](artifacts/'repository-retention-20260923/closed-september27-selection-20260928.json',
    dict(selected=selected, excluded=excluded, initial_failure=worker['pin'](failure), selector=worker['pin'](Path(__file__))))
worker['PATTERNS'] = selected
worker['__file__'] = __file__

if __name__ == '__main__': worker['main']()
