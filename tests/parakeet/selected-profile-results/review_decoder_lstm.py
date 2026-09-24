"""Retain a local-only, identity-bound review of decoder recurrent mechanisms."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-decoder-lstm-source-review-20260924'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def main():
    result_path = OUTPUT / 'decoder-lstm-source-observations-20260924.json'
    assert not BASE.exists() and not result_path.exists()
    selected = read(ROOT / 'artifacts/parakeet-inclusive-packing-source-20260924/prepared.json')['before']
    names = ['CPUExecutionProvider.Recurrent.cs', 'CPUExecutionProvider.LstmPanels.cs',
             'GraphPacking.cs', 'GraphConvPacking.cs', 'ComputationalGraph.cs',
             'GraphExecution.cs', 'TensorExecutionOptions.cs']
    sources = {}
    for name in names:
        key = 'src/Lokad.Onnx/' + name
        sources[key] = pin(ROOT / key); assert sources[key] == selected[key]
    panels = (ROOT / 'src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs').read_text()
    assert 'sequence < 8 || hiddenSize < 16 || hiddenSize > 128' in panels
    assert 'Vector.Add(a, Vector.Multiply(' in panels
    provenance = ROOT / 'artifacts/pyannote-lstm-default-gates-20260923/prepared.json'
    ort = {}
    for row in read(provenance)['ort']:
        assert row['revision'] == '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
        path = provenance.parent / ('ort-' + Path(row['path']).name)
        assert pin(path) == row['file']
        ort[path.relative_to(ROOT).as_posix()] = row
    body = (provenance.parent / 'ort-uni_directional_lstm.cc').read_text()
    assert 'float beta = 0.0f;' in body and 'beta = 1.0f;' in body
    BASE.mkdir()
    patches = {}
    for revision in ['c51c2bc271c5b33f4dab4159401afb0401393c26',
        '03af0bf3457697fd0b73c851fd7913455579604a', 'c762e7b49b6f5c8457bf47a0197f49fdd02135c7',
        '4f3a43e5c478f0a40b929bf57b7fc8229e50108f', '9e4c829f3c19739004652982caa4767a297c0a89',
        '6f25493eeeb40c619ca135776dcc6621e3806129']:
        value = subprocess.run(['git', 'show', '--format=fuller', '--no-ext-diff', revision],
                               cwd=ROOT, capture_output=True, check=True).stdout
        path = BASE / (revision + '.patch'); path.write_bytes(value)
        patches[revision] = dict(path=path.relative_to(ROOT).as_posix(), file=pin(path))
    value = dict(passed=True, selected_source=sources, ort=ort, ort_provenance=pin(provenance),
        voice_branch_patches=patches, profile_closure=pin(ROOT / 'artifacts/parakeet-selected-profile-amd-20260924/closed.json'),
        report=pin(OUTPUT / 'decoder-lstm-source-review-20260924.md'), generator=pin(Path(__file__)),
        product_changed=False, performance_measurement=False)
    result_path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(dict(passed=True, result=pin(result_path), patches=len(patches))))


if __name__ == '__main__': main()
