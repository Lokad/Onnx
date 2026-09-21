"""Preserve the failed URI assumption, then audit the unchanged completed capture."""
from common import *

path = TOOLS / 'audit_toy.py'
source = path.read_text(encoding='utf8')
assert source.count('from stacks import inspect, cross_export') == 1
source = source.replace('from stacks import inspect, cross_export', 'from stacks_v2 import inspect, cross_export')

if __name__ == '__main__':
    receipt = BASE / 'toy-audit-schema-failure.json'
    assert not receipt.exists()
    actual = read(BASE / 'toy-output/speedscope.speedscope.json')['$schema']
    assert actual == 'https://www.speedscope.app/file-format-schema.json' and not actual.endswith('speedscope/file-format-schema.json')
    save(receipt, dict(passed=False, expected_predicate="schema.endswith('speedscope/file-format-schema.json')", actual=actual,
        original_tools={rel(p): pin(p) for p in [path, TOOLS / 'stacks.py']},
        reason='Original auditor rejected the schema URI before parsing stacks; correct the literal, preserve the capture and all accounting checks.'))
    exec(compile(source, str(path), 'exec'), dict(__name__='__main__', __file__=str(path)))
