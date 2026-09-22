"""Retain complete local graph/public/shared qualification observations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def main():
    path = TOOLS/'observations-20260922.json'; assert not path.exists()
    folders = dict(pyannote=ROOT/'artifacts/pyannote-blocked-spatial-models-20260922',
                   shared=ROOT/'artifacts/pyannote-blocked-spatial-shared-20260922')
    shas = dict(pyannote='2cb2905c448861e7d4554a41c436e7eeccfea73a9f5517ea7370d9bb04fcf005',
                shared='e27a6e668548db02b77f0b0bd6cbd8f6bdced3f260faa6ad39be294e3886fe31')
    closures = {}; observations = {}
    for name, folder in folders.items():
        assert pin(folder/'closed.json')['sha256'] == shas[name]
        proof = read(folder/'closed.json'); assert proof['passed']
        for relative, wanted in proof['files'].items(): assert pin(ROOT/relative) == wanted, relative
        closures[name] = dict(path=(folder/'closed.json').relative_to(ROOT).as_posix(), **pin(folder/'closed.json'))
        observations[name] = read(folder/'analysis.json')
    observations['complete_consumer_review'] = read(folders['pyannote']/'complete.json')
    observations['graph_and_public_results'] = read(folders['pyannote']/'output/result.json')
    observations['shared_results'] = {mode: read(folders['shared']/'outputs'/mode/'result.json') for mode in ['shared', 'e5']}
    output = dict(passed=True, scope='Local complete Pyannote graph/public and native shared-model qualification',
        selected_for_production=False, amd_qualified=False, performance_comparison=False,
        closures=closures, observations=observations)
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(output=pin(path), closures=closures)))


if __name__ == '__main__': main()
