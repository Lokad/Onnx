"""Freeze a single copy candidate only after complete actual-layout qualification."""
import difflib
import json
from pathlib import Path
from run import ROOT, TOOLS, BASE, read, pin, write

SOURCE = ROOT/'artifacts/parakeet-slice-materialization-source-v2-20260924'


def main():
    failure_path = ROOT/'artifacts/parakeet-slice-materialization-build-amd-20260924/closed.json'
    failure = read(failure_path)
    assert failure['terminal'] and not failure['passed'] and len(failure['failed_tests']) == 7
    assert failure['all_copy_cases_passed'] and failure['identity_passed']
    closure = read(BASE/'closed.json'); assert closure['passed'] and closure['analysis'] == pin(BASE/'analysis.json')
    layout = read(BASE/'analysis.json')
    assert layout['layouts'] == 1920 and layout['all_parents_dense_row_major'] and layout['all_copy_regions_in_bounds']
    selected_path = ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923/prepared.json'
    selected = read(selected_path)
    for name,wanted in selected['source'].items(): assert pin(ROOT/name) == wanted, name
    assert not SOURCE.exists(); SOURCE.mkdir(); snapshot = SOURCE/'source'; snapshot.mkdir()
    original = (ROOT/'src/Lokad.Onnx/TensorSlice.cs').read_text(encoding='utf8')
    needle = '    public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions) => Clone().Reshape(dimensions);'
    assert original.count(needle) == 1
    changed = original.replace(needle,(TOOLS/'reshape.cs.txt').read_text(encoding='utf8').rstrip())
    for name in selected['source']:
        target = snapshot/name; target.parent.mkdir(parents=True,exist_ok=True)
        if name == 'src/Lokad.Onnx/TensorSlice.cs': target.write_text(changed,encoding='utf8')
        else: target.write_bytes((ROOT/name).read_bytes())
    test_name = 'tests/Lokad.Onnx.Tensors.Tests/SliceReshapeCopyTests.cs'
    (snapshot/test_name).write_bytes((TOOLS/'SliceReshapeCopyTests.cs.txt').read_bytes())
    patch = ''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),
        fromfile='a/src/Lokad.Onnx/TensorSlice.cs',tofile='b/src/Lokad.Onnx/TensorSlice.cs'))
    (SOURCE/'candidate.patch').write_text(patch,encoding='utf8')
    identities = {p.relative_to(snapshot).as_posix():pin(p) for p in snapshot.rglob('*') if p.is_file()}
    assert len(identities) == len(selected['source'])+1
    modified = [name for name in selected['source'] if identities[name] != selected['source'][name]]
    assert modified == ['src/Lokad.Onnx/TensorSlice.cs']
    value = dict(passed=True,before=selected['source'],source=identities,layout_closure=pin(BASE/'closed.json'),
        selected_source=pin(selected_path),modified=modified,added=[test_name],patch=pin(SOURCE/'candidate.patch'),
        implementation=pin(TOOLS/'reshape.cs.txt'),tests=pin(TOOLS/'SliceReshapeCopyTests.cs.txt'),
        generator=pin(__file__),root_product_changed=False,expected_upper_bound_seconds=2.937944922333333,
        previous_failure=pin(failure_path),correction='Use nullable out result with NotNullWhen instead of a null-forgiving operator')
    write(SOURCE/'prepared.json',value)
    print(json.dumps(dict(passed=True,source_files=len(identities),prepared=pin(SOURCE/'prepared.json'),modified=modified,added=value['added'])))


if __name__ == '__main__': main()
