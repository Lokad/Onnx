"""Retire the closed current Pad build's regenerable private package cache."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('padding_cache_retirement',HERE/'retire_padding_package_caches.py')
retirement=importlib.util.module_from_spec(spec);spec.loader.exec_module(retirement)
retirement.BASE=retirement.ROOT/'artifacts/parakeet-pad-current-cache-retention-20260926'
retirement.NAMES=['lokad-parakeet-pad-current-build-20260926']
retirement.PROOFS=[('parakeet-pad-current-build-amd-20260926',
                   '1835c79eda505c056cf796702ca734b6e43c65e066b3e54bb566ac25885c4018')]

if __name__=='__main__':
    # The inherited retiree checks all payload inputs. Also reject a dependency
    # in a staged-but-not-prepared or otherwise retained spec before invoking it.
    guard=retirement.run.PRELUDE+'''
from protocol import read
target=Path('/dev/shm/lokad-parakeet-pad-current-build-20260926/packages')
assert target.resolve()==target and target.parent.parent==Path('/dev/shm')
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['stage.json','spec.json']:
  p=folder/name
  if not p.is_file():continue
  value=read(p);names=[str(folder/n) for n in value.get('files',{})]+list(value.get('external',{}))
  names += [v['source'] for v in value.get('links',{}).values() if isinstance(v,dict) and isinstance(v.get('source'),str)]
  assert not any(Path(n).resolve().is_relative_to(target) for n in names),str(p)
print('stage/spec dependencies clear')
'''
    print(retirement.run.ssh(guard,300))
    retirement.main()
    retirement.save(retirement.BASE/'configuration.json',dict(adapter=retirement.pin(Path(__file__)),
        implementation=retirement.pin(HERE/'retire_padding_package_caches.py'),roots=retirement.NAMES,
        proofs=retirement.PROOFS,closed=retirement.pin(retirement.BASE/'closed.json')))
