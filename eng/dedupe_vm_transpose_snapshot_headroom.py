"""Include the newly completed attention snapshot store in verified disk deduplication."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).with_name('dedupe_vm_closed_snapshot_stores.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '98644a8d4898eb795713bf6e69893f948d3e7934aeadd24ff69f64da3674eb06'
loader = importlib.util.spec_from_file_location('closed_disk_deduplication', SOURCE)
worker = importlib.util.module_from_spec(loader); loader.loader.exec_module(worker)
worker.OUT = worker.ROOT/'artifacts/vm-transpose-snapshot-headroom-20260928'
worker.STORES += [('vm-attention-closed-evidence-offload-20260928', 'attention-closed-evidence-20260928')]
original_script = worker.script


def script():
    value = original_script()
    before = 'closed-snapshot-dedup-20260928.jsonl'
    assert value.count(before) == 1
    return value.replace(before, 'transpose-snapshot-headroom-20260928.jsonl')


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['plan', 'apply']
    code = script(); identity = hashlib.sha256(code.encode()).hexdigest()
    if sys.argv[1] == 'plan':
        assert not worker.OUT.exists()
        result = json.loads(worker.engine.run.ssh('ACTION="plan"\n'+code, 180))
        worker.OUT.mkdir()
        worker.engine.save(worker.OUT/'plan.json', dict(script_sha256=identity, **result))
        print(json.dumps({k:v for k,v in result.items() if k != 'changes'}))
    else:
        assert not (worker.OUT/'started.json').exists()
        plan = json.loads((worker.OUT/'plan.json').read_text())
        assert plan['script_sha256'] == identity
        worker.engine.save(worker.OUT/'started.json', dict(script_sha256=identity))
        result = json.loads(worker.engine.run.ssh('ACTION="apply"\nPLAN='+repr(plan)+'\n'+code, 300))
        worker.engine.save(worker.OUT/'completed.json', result)
        print(json.dumps(result))
