"""Reuse the original qualification procedure with one test-only source repair."""
import ast
from source_scope import ROOT, TOOLS, PARENT, APP, PREVIOUS, FAILED, load
from protocol import pin, read


def body(path, name):
    source = path.read_text()
    return next(ast.get_source_segment(source, node) for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def verify_scope():
    frozen = read(FAILED/'prepared.json')['files']
    for path in PREVIOUS.iterdir():
        if path.is_file(): assert pin(path) == frozen[path.relative_to(ROOT).as_posix()]
    old = load('attention_original_consumer_scope', PREVIOUS/'consumer_scope.py')
    old.TOOLS = PREVIOUS
    files = old.verify_scope()
    for name in ['checks.py', 'new_cases.py', 'warning_census.py', 'remote_prepare.py', 'audit.py', 'test_qualification.py']:
        assert (TOOLS/name).read_bytes() == (PREVIOUS/name).read_bytes(), name
    expected = body(PREVIOUS/'prepare.py', 'prepare')
    changes = [
        ("'graph_prerequisite.py','model_prerequisites.py','prerequisites.py','application_admission.py']:",
         "'graph_prerequisite.py','model_prerequisites.py','prerequisites.py','application_admission.py','fixture_repair.py']:"),
        ("['remote_prepare.py','checks.py','new_cases.py','prerequisites.py']",
         "['remote_prepare.py','checks.py','new_cases.py','prerequisites.py','fixture_repair.py']"),
        ("    copy(FIXTURE,'evidence/OwnedAttentionPreparationTests.cs')",
         "    copy(FIXTURE,'evidence/OwnedAttentionPreparationTests.cs')\n"
         "    copy(ORIGINAL_FIXTURE,'evidence/OwnedAttentionPreparationTests.original.cs.txt')\n"
         "    copy(FAILED/'failed.json','evidence/failed-root.json')\n"
         "    copy(PREVIOUS_APPLIED/'applied.json','evidence/prior-root-applied.json')"),
        ('source policy unchanged.', 'helper arguments made explicit; source policy unchanged.'),
    ]
    for before, after in changes:
        assert expected.count(before) == 1, before
        expected = expected.replace(before, after)
    assert body(TOOLS/'prepare.py', 'prepare') == expected
    for name in ['evidence_spec', 'gates', 'previous_closed']:
        assert body(TOOLS/'prepare.py', name) == body(PREVIOUS/'prepare.py', name)
    assert body(TOOLS/'prerequisites.py', 'validate') == body(PREVIOUS/'prerequisites.py', 'validate')
    assert not any((TOOLS/name).exists() for name in ['remote.py', 'protocol.py', 'application_admission.py', 'graph_prerequisite.py'])
    files.update({p.relative_to(ROOT).as_posix(): pin(p) for p in PREVIOUS.iterdir() if p.is_file()})
    return files


if __name__ == '__main__': print(dict(passed=True, frozen_files=len(verify_scope()), jobs=18))
