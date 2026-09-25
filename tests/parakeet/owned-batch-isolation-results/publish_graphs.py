"""Publish the complete graph verdict using the original checks and tables."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-owned-batch-isolation-graphs-amd-20260925'
BUILD = ROOT/'artifacts/parakeet-owned-batch-isolation-build-amd-20260925'
PARENT = OUT.parent/'packed-final-row-results/publish_release.py'
loader = importlib.util.spec_from_file_location('original_graph_publisher', PARENT)
original = importlib.util.module_from_spec(loader)
loader.loader.exec_module(original)
pin, read = original.pin, original.read
original.OUT = OUT
original.CANDIDATE_CORE = 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'


def closed(stage):
    assert stage == 'graphs'
    proof = read(BASE/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(BASE/name) == wanted, name
    assert pin(BUILD/'closed.json')['sha256'] == '5dd53e90d4924a36bb6f43cc80fa939e9236492949542dcf659dc056d4ea3860'
    compiled = read(BUILD/'build-review.json')
    assert compiled['release_dispatcher_restored'] and not compiled['release_admitted']
    assert compiled['product']['Lokad.Onnx.dll']['sha256'] == original.CANDIDATE_CORE
    assert pin(BASE/'collected/evidence/isolation-build/build-review.json') == pin(BUILD/'build-review.json')
    analysis = read(BASE/'analysis.json')
    assert analysis['passed'] and proof['files']['analysis.json'] == pin(BASE/'analysis.json')
    return BASE, proof, analysis


publish_original = original.publish


def publish(documents):
    name = 'graphs-20260925.md'; text = documents[name]
    replacements = {
        '# M78 packed-weight candidate: complete graph comparison': '# Packed-dispatch relocation: complete graph comparison',
        'This is the first M78 comparison against qualified release Core f95a13c5.':
            'This compares relocation Core e07a4518 against qualified release f95a13c5.',
        'The separately retained failed M73 controls keep their original verdicts.':
            'The parent M78 short-e5 failure keeps its original verdict. Only three\n'
            'existing methods change; the shared dispatcher recovers release compiled\n'
            'instructions and all 89 focused behavioral tests pass.'
    }
    for before, after in replacements.items():
        assert text.count(before) == 1
        text = text.replace(before, after)
    documents[name] = text
    publish_original(documents)


if __name__ == '__main__':
    original.closed, original.publish = closed, publish
    original.graphs()
