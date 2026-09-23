"""Resume extracted full-export stage after import-cache failure, before workers."""
import full_export as export
original=export.capture_run.ssh
def resumed(script,timeout=180):
 if '\nimport base64,io\n' in script:
  marker='from protocol import read,pin\n';assert script.count(marker)==1
  remainder=script.split(marker,1)[1]
  script=export.capture_run.PRELUDE+'''import importlib
importlib.invalidate_caches()
assert base.is_dir() and not any((base/n).exists() for n in ['payload.json','deployment.json','identity.json','staged.json'])
from protocol import read,pin
'''+remainder
 return original(script,timeout)
export.capture_run.ssh=resumed
export.stage()
