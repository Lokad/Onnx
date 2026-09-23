"""Publish closed build and numerical evidence without executing a workload."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BUILD=ROOT/'artifacts/pyannote-winograd-output-blocks-build-amd-20260923'
NUM=ROOT/'artifacts/pyannote-winograd-output-blocks-numerics-amd-20260923'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,v):
    assert not p.exists();p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    for base,digest in [(BUILD,'cc8d46c59b8ee1328f99c781b9fdc4f2f3ae0ff0d1592fef752f888a250558dc'),
                        (NUM,'47513c48326d17f7188a9dbf45bbd7232d4ee55122f82a3c4db23e8d12c8906b')]:
        assert pin(base/'closed.json')['sha256']==digest
        proof=read(base/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(base/name)==wanted,name
    build=read(BUILD/'analysis.json');analysis=read(NUM/'analysis.json')
    assert analysis['numerically_admitted'] and analysis['no_performance_measurement']
    actual={name:read(NUM/'collected'/name/'result.json') for name in analysis['results']}
    for mode in ['raw','captured']:
        reference=actual['current-'+mode+'-256']['rows']
        for role in ['current','candidate']:
            for width in [256,512]:assert actual[f'{role}-{mode}-{width}']['rows']==reference
        save(OUT/(mode+'-rows-20260923.json'),reference)
    observations=dict(closure=pin(NUM/'closed.json'),analysis=analysis,
        runs={name:dict(result=pin(NUM/'collected'/name/'result.json'),
                       metadata={k:v for k,v in value.items() if k!='rows'},
                       canonical_rows=mode+'-rows-20260923.json')
              for name,value in actual.items() for mode in [value['mode']]},
        artifacts={name:pin(NUM/name) for name in ['prepared.json','staged.json','payload.json','deployment.json',
            'payload.tar.gz','results.tar.gz','collection-transfer.json','storage-recovery.json']})
    save(OUT/'numerics-observations-20260923.json',observations)
    save(OUT/'build-observations-20260923.json',dict(closure=pin(BUILD/'closed.json'),analysis=build,
        artifacts={name:pin(BUILD/name) for name in ['prepared.json','staged.json','payload.json','deployment.json',
            'payload.tar.gz','results.tar.gz','collection-transfer.json']}))
    text=f'''# Two-output-block Winograd numerical qualification — 2026-09-23

**Passed; performance is not yet measured.** One common consumer binds the actual
selected Core521bae17 and candidate Core90b2164b using typed delegates. No kernel
source is compiled into that consumer. Both products return bit-identical full
rows in all four role/width combinations, including the expanded tail cases.

| Coverage per product and width | Cases | Output values | Worst Winograd scaled error |
|---|---:|---:|---:|
| Synthetic independent double reference | 3,456 | 45,958,656 | 0 |
| Captured native ORT references | 87 | 98,734,080 | 1.8477439880371094e-6 |
| Independent range-guard comparisons | 10,566 | — | Exact Boolean agreement |

The original 1,920 synthetic rows and every captured row also match the closed
M33 qualification exactly. Added cases cover input channels16/32, output channels
32/48/64/80/96/128/256/512 and shapes1x1/3x17/5x33/6x34, with all three original
patterns and eight epilogues. Outputs48/80 exercise a remaining single block.
Existing admission refuses16 output channels; two new checks prove preparation
and execution refuse that shape before mutating the supplied buffers.

All21 existing numerical/budget refusals and24 alias/extent checks per product
and width pass. Read-only operands, held/repeated outputs and buffer guards pass.
The native bound remains1e-4; the unchanged direct control's worst scaled error
is3.814697265625e-6. Zero synthetic error is specific to these dyadic patterns;
it is not a general exact-arithmetic claim.

All11 serial build/numerical jobs and{analysis['resources']} resource samples pass,
peak owned RSS{analysis['peak_rss']:,}bytes. The21 audit tests include missing and
duplicate cases, changed output/statistics, invalid error claims, refusal counts
and wrong process/product identities. SDK10.0.204/runtime10.0.8; CPU2 workers,
CPU0 monitor. Owner802237/birth1790148386.22 and every worker are terminal0.

Numerical closure `{pin(NUM/'closed.json')['sha256']}`.
Consumer `{analysis['consumer']['sha256']}`.
[Structured evidence](numerics-observations-20260923.json) preserves all role
identities and resource conclusions. [Synthetic rows](raw-rows-20260923.json)
and [captured rows](captured-rows-20260923.json) publish each common row once;
the publisher verifies complete row equality across both roles and widths.
All eight original result files remain in the closed artifact, with their
individual digests in the structured evidence. The
[consumer README](../winograd-output-blocks-prototype/README.md) gives reproduction.

The normal product build closes `{pin(BUILD/'closed.json')['sha256']}`.
Only private MultiplyWinograd512 changes; all3,178 other Core methods,697 Data
methods and public declarations match. There are no added, removed or renamed
methods. All75 build-resource observations pass, peak499,093,504bytes. The
[build observations](build-observations-20260923.json) retain scope and binaries.
The actual selected root remains unchanged.

Idle storage recovery retained verified local restore archives and checked every
remote payload dependency before reclaiming closed outputs/caches. An initial
old-work archive-coverage check refused without deleting anything; its missing
restore archive was then created and verified. All receipts remain indexed in
the closed numerical artifact. Original memory/resource bounds were preserved.

Generated-code review must pass before a separate complete-call timing screen.
Current application/ORT ratios in BENCHMARK.md remain unchanged. The
[ORT source review](../integration-review/input-broadcast-reuse-20260923.md)
explains the general broadcast-reuse mechanism and its limits.
'''
    target=OUT/'numerics-20260923.md';assert not target.exists();target.write_text(text,encoding='utf8')
    print(json.dumps(dict(report=pin(target),consumer=analysis['consumer'],resources=analysis['resources'])))


if __name__=='__main__':main()
