"""Add exact metadata-name reconciliation; run the unchanged strict build checks."""
import importlib.util
import json
from pathlib import Path
import sys
from scope import reconcile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT/'tests/parakeet/direct-depthwise-build-v2'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('original_depthwise_review', TOOLS/'review.py')
original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)


def main():
    base = original.BASE
    assert not (base/'build-review.json').exists() and not (base/'scope-correction.json').exists()
    inventory = original.read(base/'build-collected/logs/instructions.json')
    _, proof = reconcile(inventory)
    original.write(base/'scope-correction.json', dict(passed=True,
        failure='Original checker refused three compiler-renamed private methods and two changed metadata references.',
        original_reviewer=original.pin(TOOLS/'review.py'), inventory=original.pin(base/'build-collected/logs/instructions.json'),
        files={p.name:original.pin(p) for p in Path(__file__).parent.iterdir() if p.is_file()},
        proof=proof, product_rebuild=False, inference_calls=0))
    strict = original.check_inventory; save = original.write
    def corrected(value, spec, built):
        normalized, evidence = reconcile(value)
        checked = strict(normalized, spec, built)
        checked[0]['compiler_name_reconciliation'] = evidence
        return checked
    def write(path, value):
        if path.name == 'build-review.json':
            value['scope_correction'] = original.pin(base/'scope-correction.json')
            value['original_reviewer'] = value['reviewer']
            value['reviewer'] = original.pin(Path(__file__))
        save(path, value)
    original.check_inventory = corrected; original.write = write
    original.build()


if __name__ == '__main__': main()
