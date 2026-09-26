"""Keep the original numerical checks, monitor and resource policy intact."""
import ast
from pathlib import Path
from protocol import pin

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'packed-final-row-pyannote-amd'
IDENTITY = ROOT/'tests/pyannote/parameterized-qualification-consumer'


def tree(text):
    return ast.dump(ast.parse(text),include_attributes=False)


def except_functions(path, names):
    value = ast.parse(path.read_text(encoding='utf8'))
    value.body = [n for n in value.body if not isinstance(n,ast.FunctionDef) or n.name not in names]
    # Module descriptions may change; all executable statements are compared.
    value.body = [n for n in value.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    return ast.dump(value,include_attributes=False)


def verify_scope():
    files = {}
    for name in ['candidate_protocol.py','qualify_outputs.py','test_semantics.py']:
        assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(),name
    assert (TOOLS/'identity_scope.py').read_bytes() == (IDENTITY/'scope.py').read_bytes()
    expected = (PARENT/'protocol.py').read_text(encoding='utf8')
    expected = expected.replace("'consumer-inventory','selected'", "'consumer-inventory','identity-probes','selected'")
    # The only executable protocol change is insertion of the guard probes.
    before, after = ast.parse(expected), ast.parse((TOOLS/'protocol.py').read_text(encoding='utf8'))
    assert ast.dump(ast.Module(body=before.body[1:],type_ignores=[])) == ast.dump(ast.Module(body=after.body[1:],type_ignores=[]))
    assert except_functions(TOOLS/'checks.py',{'inventory'}) == except_functions(PARENT/'checks.py',{'inventory'})
    assert except_functions(TOOLS/'remote.py',{'command_for','after'}) == except_functions(PARENT/'remote.py',{'command_for','after'})
    expected = (PARENT/'audit.py').read_text(encoding='utf8')
    changes = [
        ('from prepare import APP_PAYLOAD,MODEL,OLD_DATA,NEW_DATA',
         'from prepare import APP_PAYLOAD,MODEL\nfrom identity_scope import source\nfrom identity_probes import review as review_probes'),
        ('M78 qualified release Coref95a13c5/Dataa893952f','Qualified current root'),
        ('M78 packed final row Core49901366/Data01e9e784','Current-root padding dispatcher'),
        ("(MODEL/'consumer/Program.cs').read_bytes().replace(OLD_DATA.encode(),NEW_DATA.encode())",
         "source((MODEL/'consumer/Program.cs').read_bytes())"),
        ("name.startswith('built/')", "name.startswith(('built/','runtimes/'))"),
        ('    results={}\n', "    guards=review_probes(collected,payload);assert guards==read(collected/'identity-probes/review.json')\n"
         "    assert receipt['probe_identities']==[r['child'] for r in read(collected/'identity-probes/probes.json')]\n    results={}\n"),
        ("consumers=dict(payload['consumers'],candidate=built['consumer'])", "consumers={role:built['consumer'] for role in ['selected','candidate']}"),
        ('inventory=il,results=results', 'inventory=il,identity_guards=guards,consumer_rebuilt=True,results=results')]
    for old,new in changes:
        assert expected.count(old) == 1,old
        expected = expected.replace(old,new)
    assert tree((TOOLS/'audit.py').read_text(encoding='utf8')) == tree(expected)
    expected = (PARENT/'run.py').read_text(encoding='utf8')
    changes = [
        ('parakeet-packed-final-row-pyannote-amd-20260925','parakeet-pad-current-pyannote-amd-20260926'),
        ('parakeet-packed-final-row-pyannote-20260925','parakeet-pad-current-pyannote-20260926'),
        ('def observe():\n', "def observe():\n    assert not (BASE/'closed.json').exists(), 'Preserve the closed campaign'\n"),
        ("assert state['complete'] and not any(live(i) for i in ids)\n",
         "assert state['complete'] and not any(live(i) for i in ids)\n"
         "probe_ids=[r['child'] for r in read(base/'identity-probes/probes.json')] if (base/'identity-probes/probes.json').exists() else []\n"
         "assert not any(live(i) for i in probe_ids)\n"),
        ("[*JOBS,'logs','built','evidence'", "[*JOBS,'logs','built','runtimes','reference-runtime','evidence'"),
        ("terminal=True,identities=ids,code=state['code']", "terminal=True,identities=ids,probe_identities=probe_ids,code=state['code']")]
    for old,new in changes:
        assert expected.count(old) == 1,old
        expected = expected.replace(old,new)
    assert tree((TOOLS/'run.py').read_text(encoding='utf8')) == tree(expected)
    for name in ['protocol.py','checks.py','remote.py','audit.py','run.py',
                 'candidate_protocol.py','qualify_outputs.py','test_semantics.py']:
        files[(PARENT/name).relative_to(ROOT).as_posix()] = pin(PARENT/name)
    for name in ['scope.py','test_scope.py']:
        files[(IDENTITY/name).relative_to(ROOT).as_posix()] = pin(IDENTITY/name)
    return files


if __name__ == '__main__':
    print(verify_scope())
