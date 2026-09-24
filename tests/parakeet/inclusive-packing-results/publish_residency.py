"""Publish every retained mapping from the closed actual-model residency audit."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-inclusive-packing-residency-amd-v2-20260924'
OUTPUT = Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    assert pin(BASE / 'closed.json')['sha256'] == '82342a58fddab63974e8bff009916a91376beb1d91f9db264480c2763c8e7b0a'
    proof = json.loads((BASE / 'closed.json').read_text()); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(BASE / name) == wanted, name
    analysis = json.loads((BASE / 'analysis.json').read_text())
    assert analysis['passed'] and analysis['model_loads'] == 8 and analysis['comparison']['both_modes_exact']
    rows = []; inputs = {}
    for name in analysis['residency']:
        path = BASE / 'collected' / name / 'result.json'
        value = json.loads(path.read_text()); inputs[path.relative_to(ROOT).as_posix()] = pin(path)
        role, graph, mode = name.split('-')
        changes = analysis['comparison']['graphs'][graph]
        for weight in value['weights']:
            status = 'shared' if weight['name'] in changes['retained'] else ('displaced' if role == 'selected' else 'newly retained')
            rows.append(dict(role=role, graph=graph, avx512=mode == '512', status=status, name=weight['name'],
                packed_name=weight['packed_name'], source_initializer_keys=';'.join(weight['source_initializer_keys']),
                reduction=weight['shape'][0], columns=weight['shape'][1], bytes=weight['bytes'],
                source_sha256=weight['source_sha256'], packed_sha256=weight['packed_sha256']))
    assert len(rows) == 142
    files = [OUTPUT / ('residency-' + suffix) for suffix in ['weights-20260924.csv', '20260924.md', 'observations-20260924.json']]
    assert not any(p.exists() for p in files)
    with files[0].open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    samples = sum(r['samples'] for r in analysis['resources']); peak = max(r['peak_rss'] for r in analysis['resources'])
    assert samples == 60 and peak == 3159846912
    text = '''# Inclusive packing: actual Parakeet residency

The candidate admits six 4096-row encoder projections while keeping the encoder
cap at 256 MiB. They displace fifteen previously retained weights: 22 mappings
remain shared, and the total changes from 37 to 28 weights with identical retained
bytes. The decoder's three mappings and 25,246,720 retained bytes are unchanged.

| Graph | Product | Retained weights | Retained bytes | Cap bytes | 4096-row weights |
|---|---|---:|---:|---:|---:|
| Encoder | Selected | 37 | 268,435,456 | 268,435,456 | 0 |
| Encoder | Candidate | 28 | 268,435,456 | 268,435,456 | 6 |
| Decoder | Selected | 3 | 25,246,720 | 67,108,864 | 0 |
| Decoder | Candidate | 3 | 25,246,720 | 67,108,864 | 0 |

Both instruction modes produce exactly the same mappings and source/clone hashes.
Every shared weight also matches between selected and candidate products.
All eight loads verify independent clone storage, source-array ownership,
aggregate accounting, invalidation and exact rebuilding without source mutation.
[All 142 retained-weight observations](residency-weights-20260924.csv) include
names, source keys, dimensions, byte counts, digests and displacement status.

All eleven AMD jobs and 60 resource observations pass. Peak observed owned RSS
is 3,159,846,912 bytes; this is load-only diagnostic memory, not application peak
memory. Compute uses CPU2, monitoring CPU0, SDK 10.0.204 / runtime 10.0.8.
Supervisor 912137 / birth 1790228622.0 and every child are terminal.

The initial helper build failed before loading models because three Dims accesses
used DenseTensor expressions instead of ITensor. Failure closure `1ff9d623`
preserves that source and compiler output. Corrected tools `5aa2fb4c` change
only those interface accesses and namespace paths; all assertions, products,
budgets, process order and limits remain unchanged. No product was rebuilt.

The consumer is `86574c04a8f4283aab6ef1e1860013926c593691687c57be8ec67548096f35a5`
(23,040 bytes). Artifact: `artifacts/parakeet-inclusive-packing-residency-amd-v2-20260924`.
Closure: `82342a58fddab63974e8bff009916a91376beb1d91f9db264480c2763c8e7b0a`.
Payload: `ec6105d3f4f62889a5bcfaec050f2aa819120db418b93e952a8f8acfdce71086`.

These are actual current AMD observations, independently confirming the earlier
Windows residency pattern. A retained mapping is not proof that every request
uses it. Full native/public correctness and complete application timing must
decide whether this budget redistribution is beneficial. BENCHMARK.md and the
selected release remain unchanged.
'''
    files[1].write_text(text, encoding='utf8', newline='\n')
    observations = dict(passed=True, closure=pin(BASE / 'closed.json'), analysis=pin(BASE / 'analysis.json'),
        products=analysis['identities'], consumer=analysis['consumer'], results=inputs,
        comparison=analysis['comparison'], resources=analysis['resources'],
        outputs={p.name: pin(p) for p in files[:2]}, generator=pin(Path(__file__)))
    files[2].write_text(json.dumps(observations, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(dict(passed=True, rows=len(rows), files={p.name: pin(p) for p in files})))


if __name__ == '__main__': main()
