"""Close verified AMD diarization evidence, then render its finite observations."""
from pathlib import Path
import argparse
import json
import subprocess
import time
from prepare import sha, read, pin, write_new


def close(base):
    assert not (base/'closed.json').exists()
    audit=read(base/'audit.json');assert audit['passed']
    assert audit['auditor_sha256']==sha(base/'payload/audit.py')
    assert audit['managed_sha256']==sha(base/'collected/result/managed.json')
    assert audit['bundle_sha256']==sha(base/'payload/bundle.json')
    assert audit['collection_sha256']==sha(base/'collected/collection.json')
    for name,expected in read(base/'payload/bundle.json')['files'].items():assert pin(base/'payload'/name)==expected,name
    for name,expected in read(base/'collected/collection.json')['files'].items():assert pin(base/'collected'/name)==expected,name
    script='''from pathlib import Path
import json
items=ITEMS
for item in items:
    path=Path('/proc')/str(item['pid'])/'stat'
    if path.exists():assert int(path.read_text().split(') ',1)[1].split()[19])!=item['start'],item
print(json.dumps(dict(all_absent=True,identities=items)))
'''.replace('ITEMS',repr(audit['resources']['terminal_processes']))
    result=subprocess.run(['ssh','-i','C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem','-o','BatchMode=yes',
        'vermorel@74.178.91.76','python3 -B -'],input=script,text=True,capture_output=True,check=True)
    root=Path(__file__).resolve().parents[3]
    record=dict(schema=1,closed=True,closed_at=time.time(),all_owned_processes_terminal=True,
        remote_terminal_check=json.loads(result.stdout),audit_sha256=sha(base/'audit.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()},
        tracked={p.relative_to(root).as_posix():pin(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.cs','.md')})
    write_new(base/'closed.json',record)
    print(json.dumps(dict(closed_sha256=sha(base/'closed.json'),files=len(record['files']),tracked=len(record['tracked'])),indent=2))


def report(base,destination):
    markdown=destination/'results-20260919.md';observations=destination/'observations-20260919.json'
    assert not markdown.exists() and not observations.exists()
    receipt=read(base/'closed.json');assert receipt['closed'] and receipt['all_owned_processes_terminal']
    assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==set(receipt['files'])|{'closed.json'}
    for name,expected in receipt['files'].items():assert pin(base/name)==expected,name
    value=read(base/'audit.json');assert value['passed'] and sha(base/'audit.json')==receipt['audit_sha256']
    value=dict(value,closed_sha256=sha(base/'closed.json'),reporter_sha256=sha(Path(__file__)))
    write_new(observations,value)
    r=value['resources']
    text=f'''# Maximum-duration diarization on AMD

`Community1Diarizer.Diarize` completes the retained 600-second speech request on AMD with the current qualified product DLLs. Native and retained Windows public timelines and speaker identities agree, and centroids pass the unchanged numerical gate. Input/output ownership and short-request recovery pass. This is one constructed maximum-duration observation; full intermediate tensor agreement and independent long-conversation accuracy remain open.

The input repeats the same pinned thirty-second dialogue twenty times: 9,600,000 mono 16 kHz samples and 591 overlapping model windows. The native oracle was independently reproduced earlier with all 4,146 arrays retained; preparation reverified their hashes and 260,571,414 finite values. The new AMD runner uses product `8732831`, core `05884cfd` and Data `27598aa8`. The old Windows result used product `21f3e74`; absolute timings across those builds and hosts are not a matched comparison.

| Observation | AMD result |
|---|---:|
| Maximum public request | {value['request_seconds']:.6f} seconds |
| Thirty-second recovery | {value['recovery_seconds']:.6f} seconds |
| Windows | {value['windows']} |
| Learned clusters | {value['speakers']} |
| Ordinary / exclusive intervals | {value['ordinary_intervals']} / {value['exclusive_intervals']} |
| Sampled process-group peak | {r['peak_rss']/1e9:.6f} GB |
| Process-reported peak working set | {value['process_peak_working_set']/1e9:.6f} GB |
| Minimum system available memory | {r['minimum_available_memory']/1e9:.6f} GB |
| Retained resource samples | {r['samples']} |

The AMD EPYC 9V74 worker inherits CPU 2 before startup, with supervisor CPU 0, .NET 10.0.8 and no experimental environment overrides. API times include frontend, neural inference, clustering and owned output construction; model loading and external validation are outside the stopwatch. All eight-GiB RSS, 1,800-second and 256-MiB available-memory guards pass. These are finite observations, not universal resource bounds or calibrated latency results.

The full call passes maximum centroid scaled error `{value['maximum_native_centroid_error']:.12g}` against native; short recovery passes `{value['maximum_recovery_centroid_error']:.12g}`. Both use `abs(actual-reference)/max(1,abs(reference)) <= 1e-4`. Maximum centroid difference from the retained Windows calls is `{value['maximum_windows_centroid_difference']:.12g}`. Native timeline boundaries use the existing `1e-12` tolerance; discrete speaker identities and interval coverage remain exact. Four learned clusters on repeated two-speaker speech do not establish four people. The declared stable native tie policy still differs from original native unstable selection at 98 exclusive frame values, with zero ordinary-frame differences; no policy or tolerance changed.

The runner rejects one sample beyond the limit and pre-canceled input, accepts the exact empty result, retains the full result and inputs unchanged through short recovery, and verifies all model/product identities before inference. The independent auditor checks both full and recovery outputs, process births, every resource sample and complete collection integrity. Three test methods pass, including 19 damaged application records, 12 damaged resource records and six prospective runtime guard failures. The existing filterbank and older silence numerical failures remain outside this public-application qualification.

All observed worker/supervisor identities are terminal. The one-time collection, audit and closure are complete; do not rerun successful writers. [Reproduction instructions](README.md) describe the retained runner and exact scope. Raw evidence is under `artifacts/pyannote-maximum-amd-20260919`; [all observations](observations-20260919.json) include process births and hashes.

- Closed receipt: `{value['closed_sha256']}`.
- Payload bundle: `{value['bundle_sha256']}`.
- Collection: `{value['collection_sha256']}`.
- Managed output: `{value['managed_sha256']}`.
- Original long native/application receipt: `e8c8371f887ea0b5a194d9e370dc33c164919c6362af5a0c82358a2f1bce314a`.
'''
    markdown.write_text(text,encoding='utf-8');print('Wrote closed AMD maximum-duration observations.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('close','report'));parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=Path(__file__).parent)
    args=parser.parse_args()
    if args.action=='close':close(args.artifact.resolve())
    else:report(args.artifact.resolve(),args.destination.resolve())
