"""One layout dispatch change, selected by the exact ORT transpose diagnosis."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-transpose-axis-source-20260928'
QUALIFIED = ROOT/'artifacts/parakeet-attention-owned-root-recovery-amd-20260928'
DIAGNOSIS = ROOT/'tests/parakeet/attention-owned-profile-results/transpose-breakdown-20260928.json'
TARGET = 'src/Lokad.Onnx/TensorOps.Shape.cs'
TEST = 'tests/Lokad.Onnx.Backend.Tests/TransposeAxisMovementTests.cs'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def change(raw):
    before = raw.decode().replace('\r\n', '\n')
    pairs = [
        ('    /// 4D head-merge face tiles directly; all other shapes keep the odometer over',
         '    /// axis-1-to-last float faces use matrix tiles; other shapes keep the odometer over'),
        ('''    static void TransposeInto(Tensor<T> data, DenseTensor<T> destination, int[] perm)
    {
        int rank = data.Rank;''',
         '''    static void TransposeInto(Tensor<T> data, DenseTensor<T> destination, int[] perm)
    {
        // Empty tensors have no faces; do not flatten their potentially large axes.
        if (destination.Length == 0) return;
        int rank = data.Rank;'''),
        ('''        // Fast path: the 4D head-merge face (0,2,3,1) is one small 2D rotation
        // per batch-and-token position, tiled directly instead of odometer-stepped.''',
         '''        // Moving axis 1 to the end is a matrix transpose per batch. Use the
        // existing contiguous-load tile when both axes contain a complete tile.
        if (rank == 3 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1
            && typeof(T) == typeof(float) && AblationSwitches.EnableVectorTransposeFaces && Avx.IsSupported
            && xd.Dimensions[1] >= 8 && xd.Dimensions[2] >= 8)
        {
            unsafe
            {
                using var source = xd.Buffer.Pin();
                using var target = destination.Buffer.Pin();
                transpose_unsafe_shuffle8x8_lastTwoAxes(xd.Dimensions[0], 1, xd.Dimensions[1], xd.Dimensions[2],
                    (float*)source.Pointer, (float*)target.Pointer);
            }
            return;
        }
        // [B,H,S,D] -> [B,S,D,H] collapses to [B,H,S*D] -> [B,S*D,H].
        // Nonempty dense storage ensures the flattened suffix fits in an int.'''),
        ('''                    transpose_unsafe_vector8_headMerge(dimB, dimS, dimH, dimD, (float*)source.Pointer, (float*)target.Pointer);''',
         '''                    if (dimH >= 8 && dimS * dimD >= 8)
                        transpose_unsafe_shuffle8x8_lastTwoAxes(dimB, 1, dimH, dimS * dimD, (float*)source.Pointer, (float*)target.Pointer);
                    else
                        transpose_unsafe_vector8_headMerge(dimB, dimS, dimH, dimD, (float*)source.Pointer, (float*)target.Pointer);''')]
    after = before
    for old, new in pairs:
        assert after.count(old) == 1, old
        after = after.replace(old, new)
    restored = after
    for old, new in reversed(pairs):
        assert restored.count(new) == 1
        restored = restored.replace(new, old)
    assert restored == before
    data = (after.replace('\n', '\r\n') if b'\r\n' in raw else after).encode()
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=TARGET, tofile=TARGET))
    return data, patch


def references():
    assert pin(QUALIFIED/'closed.json')['sha256'] == 'ee4a38ff671cc3fd8cd608c0dc3008f5c1b99f61ba6c139f3deeaf0e9039305e'
    closure = read(QUALIFIED/'closed.json')
    assert closure['passed'] and closure['files']['bundle/stage.json'] == pin(QUALIFIED/'bundle/stage.json')
    stage = read(QUALIFIED/'bundle/stage.json')
    source = {n.removeprefix('source/'): v for n, v in stage['files'].items() if n.startswith('source/')}
    assert len(source) == 446
    for name, wanted in source.items():
        assert pin(QUALIFIED/'bundle/source'/name) == pin(ROOT/name) == wanted, name
    diagnosis = read(DIAGNOSIS)
    assert diagnosis['passed'] and diagnosis['selected_nodes'] == 96 and not diagnosis['performance_claim']
    assert diagnosis['root'] == pin(QUALIFIED/'closed.json')
    assert diagnosis['partition'] == pin(ROOT/'artifacts/parakeet-attention-owned-gap-20260928/closed.json')
    assert diagnosis['source'] == pin(DIAGNOSIS.with_name('transpose_review.py'))
    for name, wanted in diagnosis['sources'].items():
        if name.startswith('src/'):
            assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == wanted['git_blob_sha256']
    return source


def main():
    assert not BASE.exists()
    original = references()
    values = {n: (QUALIFIED/'bundle/source'/n).read_bytes() for n in original}
    values[TARGET], patch = change(values[TARGET])
    assert TEST not in values
    values[TEST] = (TOOLS/'TransposeAxisMovementTests.cs.txt').read_bytes()
    BASE.mkdir()
    for name, data in values.items():
        path = BASE/'source'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    (BASE/'candidate.patch').write_text(patch, encoding='utf8')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'PLAN.md').read_bytes())
    source = {n: pin(BASE/'source'/n) for n in values}
    assert [n for n in original if source[n] != original[n]] == [TARGET]
    write(BASE/'prepared.json', dict(passed=True, release_admitted=False, root_product_changed=False,
        baseline=pin(QUALIFIED/'closed.json'), diagnosis=pin(DIAGNOSIS), source=source, source_before=original,
        changed_product_files=[TARGET], added_product_files=[], changed_methods=['TransposeInto'],
        added_methods=[], added_tests=[TEST], source_reversible=True, arithmetic_leaves_unchanged=True,
        prediction_corpus_seconds_saved=0.45, patch=pin(BASE/'candidate.patch'), plan=pin(BASE/'prospective-plan.md'),
        tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'), files=len(source), changed=[TARGET], added_tests=[TEST])))


if __name__ == '__main__': main()
