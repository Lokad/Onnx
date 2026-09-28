"""Update the release shortlist only from published, admitted and source-matched evidence."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import sys
from common import application, pin, read

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'tests/parakeet/transpose-axis-root-amd'))
from source_scope import CHANGED, verify_source, root_files, verify_root


def published(campaign, filename):
    base = ROOT/'artifacts'/('parakeet-transpose-axis-'+campaign+'-amd-20260928')
    proof = read(base/'closed.json'); value = read(base/'analysis.json')
    assert proof['passed'] and value['passed']
    assert pin(base/'analysis.json') == proof.get('analysis', proof['files']['analysis.json'])
    report = read(OUT/filename)
    assert report['closure'] == pin(base/'closed.json')
    assert all(report[key] == item for key, item in value.items())
    state = read(base/'collected/identity.json')
    assert state['complete'] and state['code'] == 0
    assert all(row['complete'] and row['code'] == 0 for row in state['runs'])
    return base, proof, value


def document():
    parakeet = application()
    graph_base, graph_proof, graphs = published('graphs', 'graphs-20260928.json')
    py_base, py_proof, pyannote = published('pyannote-app', 'pyannote-application-20260928.json')
    root_base, _, root = published('root', 'root-observations-20260928.json')
    assert graph_proof['admitted'] and all(row['qualified'] for row in graphs['performance'])
    assert py_proof['admitted'] and pyannote['performance']['admitted']
    assert root['root_source_verified'] and root['package']['passed'] and root['consumer']['passed']
    candidate = parakeet['identities']['candidate']
    assert root['measured'] == pyannote['identities']['candidate'] == candidate
    assert pyannote['identities']['selected'] == parakeet['identities']['current']
    assert graphs['products'] == {
        role: {'Lokad.Onnx.dll': parakeet['identities'][role]['Lokad.Onnx.dll']}
        for role in ['current', 'candidate']}
    assert root['inventory']['core_methods'] == 3288 and root['inventory']['data_methods'] == 697
    assert (root['suites']['backend']['passed'], root['suites']['backend']['skipped']) == (3603, 43)
    assert (root['suite256']['backend']['passed'], root['suite256']['backend']['skipped']) == (3513, 133)
    assert root['suites']['tensors']['passed'] == root['suite256']['tensors']['passed'] == 394
    applied = read(root_base/'bundle/evidence/root-applied.json')
    assert root['root_integration'] == pin(root_base/'bundle/evidence/root-applied.json')
    assert applied['source_files'] == root_files(verify_source())
    verify_root(applied['source_files'])
    tracked = subprocess.check_output(['git', 'ls-files', '--error-unmatch', '--', *CHANGED],
                                      cwd=ROOT, text=True).splitlines()
    assert set(tracked) == set(CHANGED)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *CHANGED], cwd=ROOT, check=True)
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', *CHANGED],
                                     cwd=ROOT, text=True).strip()
    assert len(commit) == 40

    corpus, = [row for row in parakeet['table'] if row['is_corpus']]
    dialogue, = [row for row in pyannote['table'] if row['name'] == 'dialogue-30s']
    assert corpus['audio_seconds'] == 213.265 and dialogue['audio_seconds'] == 30
    numbers = [(corpus['candidate']['seconds'], corpus['ort']['seconds']),
               (dialogue['candidate']['seconds'], dialogue['ort']['seconds'])]
    graph_rows = {row['key']: row for row in graphs['performance']}
    numbers += [(graph_rows[key]['candidate'], graph_rows[key]['ort'])
                for key in ['e5-30tok', 'dinov3', 'resnet50', 'gpt2']]
    labels = [('Parakeet TDT 0.6B V3', 'Transcribe 20 clips / 213.265 seconds of audio'),
              ('Pyannote Community-1', 'Complete diarization of a 30-second dialogue'),
              ('multilingual-e5-small', 'One 30-token forward pass'),
              ('DINOv3 ViT-S/16', 'One 224x224 image, full weights'),
              ('ResNet50', 'One 224x224 image, feature export'),
              ('GPT-2', 'Four-token prefill, empty past state')]
    rows = []
    for (model, workload), (managed, native) in zip(labels, numbers, strict=True):
        assert managed > 0 and native > 0
        rows.append(f'| {model} | {workload} | {managed:.6f} | {native:.6f} | **{managed/native:.3f}** | Qualified |')
    path = ROOT/'BENCHMARK.md'; old = path.read_text(encoding='utf8')
    assert 'source `7e321ecc`' in old, 'Require the previous qualified release document'
    start = old.index('| Parakeet TDT'); end = old.index('| DINOv2-small', start)
    value = old[:start] + '\n'.join(rows) + '\n' + old[end:]
    start = value.index('The selected product is source '); end = value.index('\n## What is timed', start)
    measured_core, measured_data = [candidate[name]['sha256'][:8] for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']]
    built_core, built_data = [root['built'][name]['sha256'][:8] for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']]
    product = f'''The selected product is source `{commit[:8]}`, measured as Core `{measured_core}`
and Data `{measured_data}`. Its [root and package qualification](tests/parakeet/transpose-axis-results/root-20260928.md)
verifies that the normal build (Core `{built_core}`, Data `{built_data}`) preserves all
3,288 Core and 697 Data method bodies, implementation flags, public declarations
and assembly attributes. Both full test suites pass in normal and AVX512-disabled
modes, and independent NuGet consumption passes. Ordinary mode passes 3,603 backend
and 394 tensor tests; the report records the exact hardware-dependent skips.
'''
    value = value[:start] + product + value[end:]
    replacements = {
        'tests/parakeet/attention-owned-results/application-20260928.md':
            'tests/parakeet/transpose-axis-results/application-20260928.md',
        'tests/parakeet/attention-owned-results/pyannote-application-20260928.md':
            'tests/parakeet/transpose-axis-results/pyannote-application-20260928.md',
        'tests/parakeet/attention-owned-results/graphs-20260928.md':
            'tests/parakeet/transpose-axis-results/graphs-20260928.md',
        'attention-owned-graphs-amd/README.md': 'transpose-axis-graphs-amd/README.md',
        'attention-owned-pyannote-app-amd/README.md': 'transpose-axis-pyannote-app-amd/README.md',
        'attention-owned-app-amd/README.md': 'transpose-axis-app-amd/README.md'}
    for before, after in replacements.items():
        assert value.count(before) == 1, before
        value = value.replace(before, after)
    # Report the actual UTC execution interval of the timing campaigns.
    dates = []
    for base in [ROOT/'artifacts/parakeet-transpose-axis-app-amd-20260928', graph_base, py_base]:
        state = read(base/'collected/identity.json')
        dates.extend(datetime.fromtimestamp(state[key], timezone.utc).date().isoformat()
                     for key in ['started', 'ended'])
    first, last = min(dates), max(dates)
    measured = first if first == last else first + ' through ' + last
    before = 'Current repository product, measured on 2026-09-27 through 2026-09-28 UTC.'
    assert value.count(before) == 1
    value = value.replace(before, 'Current repository product, measured on ' + measured + ' UTC.')
    return path, old, value, dict(source_commit=commit, measured=candidate, built=root['built'], execution_utc=[first,last],
                                  rows=[dict(model=model, managed=a, ort=b, ratio=a/b)
                                        for (model, _), (a, b) in zip(labels, numbers, strict=True)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Apply the verified document update.')
    args = parser.parse_args()
    path, old, value, report = document()
    if args.write:
        assert path.read_text(encoding='utf8') == old
        path.write_text(value, encoding='utf8', newline='\n')
    print(json.dumps(dict(passed=True, written=args.write, **report), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
