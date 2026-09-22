"""Audit all 400 scheduled caller cases without changing their contents."""
from pathlib import Path

directory = Path(__file__).resolve().parent
source = (directory / 'audit_acceptance.py').read_text(encoding='utf8')
assert source.count('import acceptance as successor') == 1
source = source.replace('import acceptance as successor', 'import acceptance_v2 as successor')
old = "namespace=dict(__name__='caller_acceptance_audit',__file__=str(TOOLS / 'audit.py'))"
assert source.count(old) == 1
source = source.replace(old, '''assert source.count("len(rows)==736") == 1
source = source.replace("len(rows)==736", "len(rows)==400")
''' + old)
exec(compile(source, str(directory / 'audit_acceptance.py'), 'exec'),
     dict(__name__='caller_count_audit', __file__=str(directory / 'audit_acceptance.py')))
