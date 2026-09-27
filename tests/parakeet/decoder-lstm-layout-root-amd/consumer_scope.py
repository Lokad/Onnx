"""Keep root workers, package checks and audit semantics bound to the qualified release."""
import ast
from source_scope import ROOT, TOOLS, PARENT, APP, load
from protocol import pin, read


def verify_scope():
    original = read(ROOT/'artifacts/parakeet-pad-current-root-amd-20260926/prepared.json')['files']
    files = {}
    for path in PARENT.iterdir():
        if path.is_file():
            name = path.relative_to(ROOT).as_posix()
            assert pin(path) == original[name], name
            files[name] = pin(path)
    application = load('lstm_layout_application_scope', APP/'consumer_scope.py')
    files.update(application.verify_scope())
    files.update({p.relative_to(ROOT).as_posix(): pin(p) for p in APP.iterdir() if p.is_file()})
    expected = (PARENT/'checks.py').read_text()
    assert expected.count('3282') == 2
    expected = expected.replace('3282', '3286').replace('(3552,42)', '(3566,42)').replace('(3462,132)', '(3476,132)')
    assert (TOOLS/'checks.py').read_text() == expected
    expected = (PARENT/'warning_census.py').read_text().replace(
        'parakeet-owned-batch-isolation-root-policy-amd-20260925', 'parakeet-decoder-packed-row-root-amd-20260927')
    assert (TOOLS/'warning_census.py').read_text() == expected
    before = (PARENT/'remote_prepare.py').read_text()
    after = (TOOLS/'remote_prepare.py').read_text()
    def main_body(text):
        node, = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main']
        return ast.get_source_segment(text, node)
    expected = main_body(before).replace("previous=read(BUILD/'payload.json')",
        "previous=read(PRIOR['qualified-root']/'payload.json')").replace(
        "shutil.copytree(BUILD/'runtime',BASE/'measured',copy_function=os.link)",
        "shutil.copytree(MEASURED,BASE/'measured',copy_function=os.link)")
    expected = expected.replace("        for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name", "        if label not in ['baseline','consumer-qualified','qualified-root']:\n            for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name")
    assert main_body(after) == expected
    for path in [TOOLS.parent/'decoder-lstm-layout-integration/prepare_tests.py',
                 TOOLS.parent/'owned-batch-isolation-build/Bridge.cs.txt',
                 TOOLS.parent/'selected-profile-build-amd/Bridge.csproj']:
        files[path.relative_to(ROOT).as_posix()] = pin(path)
    assert not any((TOOLS/name).exists() for name in ['remote.py','protocol.py','application_admission.py','graph_prerequisite.py'])
    return files


if __name__ == '__main__': print(dict(passed=True, frozen_files=len(verify_scope()), jobs=18))
