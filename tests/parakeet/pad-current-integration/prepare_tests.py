"""Prepare the tested Pad fixture for the repository's explicit-argument policy."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'artifacts/parakeet-pad-current-source-20260926'
BASE=ROOT/'artifacts/parakeet-pad-integration-tests-20260926'
TEST='tests/Lokad.Onnx.Backend.Tests/LastAxisPadTests.cs'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    assert not BASE.exists() and not (ROOT/TEST).exists()
    original=SOURCE/'source'/TEST
    prepared=json.loads((SOURCE/'prepared.json').read_text())
    assert pin(original)==prepared['source'][TEST]
    before=original.read_bytes();after=before
    edits=[('T fill, bool reversed = false)','T fill, bool reversed)'),
           ('Check(logical, shape, pads, fill);','Check(logical, shape, pads, fill, false);'),
           ('Check(Array.Empty<T>(), new[] { 2, 0, 3 }, new[] { 0, 0, 4, 0, 0, 4 }, fill);',
            'Check(Array.Empty<T>(), new[] { 2, 0, 3 }, new[] { 0, 0, 4, 0, 0, 4 }, fill, false);'),
           ('Check(data, new[] { 2, 3, 4 }, new[] { 1, -1, -1, 0, 1, 2 }, fill);',
            'Check(data, new[] { 2, 3, 4 }, new[] { 1, -1, -1, 0, 1, 2 }, fill, false);')]
    for old,new in edits:
        assert after.count(old.encode())==1
        after=after.replace(old.encode(),new.encode())
    assert b'bool reversed = false' not in after and after.count(b'[Fact]')==6
    restored=after
    for old,new in reversed(edits):restored=restored.replace(new.encode(),old.encode())
    assert restored==before
    BASE.mkdir();target=BASE/'LastAxisPadTests.cs';target.write_bytes(after)
    patch=''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),
        fromfile='measured/'+TEST,tofile='integration/'+TEST))
    (BASE/'review.patch').write_text(patch,encoding='utf8')
    result=dict(passed=True,source_prepared=pin(SOURCE/'prepared.json'),original=pin(original),corrected=pin(target),
        patch=pin(BASE/'review.patch'),helper_defaults_removed=1,call_arguments_made_explicit=3,
        assertions_and_test_names_unchanged=True,product_unchanged=True,root_applied=False,
        source_policy_test=pin(ROOT/'tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'),
        script=pin(Path(__file__)),full_root_suites_still_required=True)
    (BASE/'prepared.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
