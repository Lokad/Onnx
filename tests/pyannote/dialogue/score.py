"""Score fixed dialogue native/managed timelines against pinned annotations.

This reports labeled accuracy only. Run the complete pipeline auditor separately
for native-output conformance, ownership, repetitions and numerical gates.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'accuracy'))
from diarization_error import score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(corpus_path, reference_path, public_paths):
    pins = json.loads((HERE / 'pins.json').read_text(encoding='utf-8'))
    if sha(corpus_path) != pins['corpus']['manifest_sha256']:
        raise ValueError('Corpus annotation identity')
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    if reference['pins'] != pins or reference['speech_manifest_sha256'] != sha(corpus_path):
        raise ValueError('Native corpus identity')
    names = [c['name'] for c in pins['cases']]
    if [c['name'] for c in reference['cases']] != names:
        raise ValueError('Native case coverage')
    public = []
    for path in public_paths:
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['reference_sha256'] != sha(reference_path):
            raise ValueError('Managed reference identity')
        rows = [r for r in result['reports'] if r['repeat'] == 0]
        if [r['name'] for r in rows] != names:
            raise ValueError('Managed first-request case coverage')
        public.append((path, rows))
    rows = []
    for index, (truth, native) in enumerate(zip(corpus['cases'], reference['cases'], strict=True)):
        duration = truth['samples'] / 16000
        if native['seconds'] != duration:
            raise ValueError('Native duration')
        for kind, key, managed_key in [('ordinary', 'intervals', 'Intervals'), ('exclusive', 'exclusive_intervals', 'ExclusiveIntervals')]:
            hypotheses = [('native-policy', native[key])]
            for path, reports in public:
                result = reports[index]['result']
                if result['AudioDuration'] != duration:
                    raise ValueError('Managed duration')
                hypotheses.append((str(path), [[v['Start'], v['End'], v['Speaker']] for v in result[managed_key]]))
            for engine, hypothesis in hypotheses:
                rows.append(dict(case=truth['name'], kind=kind, engine=engine,
                                 metrics=score(truth['annotations'], hypothesis, duration)))
    return dict(scope='Labeled accuracy only; one recording and three correlated boundary excerpts',
                primary='dialogue-30s', corpus_sha256=sha(corpus_path), reference_sha256=sha(reference_path),
                public_sha256={str(path): sha(path) for path in public_paths},
                recipes={'score.py': sha(Path(__file__)), 'diarization_error.py': sha(HERE.parent / 'accuracy/diarization_error.py')},
                versions={n: importlib.metadata.version(n) for n in ('pyannote.metrics', 'pyannote.core', 'numpy', 'scipy', 'pandas')},
                rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Native manifest.json')
    parser.add_argument('--public', type=Path, action='append', required=True, help='Managed replay JSON; repeat for another configuration')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh output')
    result = evaluate(args.corpus, args.reference, args.public)
    with args.output.open('x', encoding='utf-8') as target:
        json.dump(result, target, indent=2)
    for row in result['rows']:
        print(row['case'], row['kind'], row['engine'], 'DER', row['metrics']['diarization_error_rate'])
