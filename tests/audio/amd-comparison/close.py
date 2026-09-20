"""Publish the audited AMD baseline without altering historical Windows results."""
from pathlib import Path
import datetime, json, shutil
from deploy import BASE,ROOT,ssh
from protocol import pin,read,write


def main():
    audit=read(BASE/'audit.json');assert audit['passed'] and audit['counts']==dict(conformance=88,timing=704)
    base=BASE/'collected';receipt=read(base/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    response=ssh('''import sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
print('terminal')
'''%receipt['births']);assert response.strip()=='terminal'
    folder=Path(__file__).resolve().parent;report=folder/'results-20260920.md';data=folder/'observations-20260920.json'
    write(data,audit)
    labels={'parakeet':'Parakeet','whisper':'Whisper Large V3 Turbo','pyannote':'pyannote'}
    header='| Application / workload | Lokad seconds | Microsoft ORT seconds | Lokad / ORT | Lokad RTF | ORT RTF |\n|---|---:|---:|---:|---:|---:|'
    headline=[];details=[];variation=[]
    for row in audit['table']:
        name=labels[row['family']]+', '+('all 20 clips (213.265 s audio)' if row['name']=='complete-corpus' else row['name'])
        line=f"| {name} | {row['managed']['seconds']:.3f} | {row['ort']['seconds']:.3f} | {row['ratio']:.3f} | {row['managed']['rtf']:.3f} | {row['ort']['rtf']:.3f} |"
        if row['name']=='complete-corpus' or row['family']=='pyannote':
            headline.append(line)
            for engine in ['managed','ort']:
                visits=row[engine]['visits'];variation.append(f"| {name} | {engine} | {visits[0]['mean']:.6f} | {visits[1]['mean']:.6f} |")
        else:details.append(line)
    table=header+'\n'+'\n'.join(headline)
    resource_rows=[]
    for row in audit['observations']:
        resource_rows.append(f"| {row['name']} | {row['resource']['samples']} | {row['resource']['peak_rss']:,} | {row['value']['setup_seconds']:.3f} |")
    frozen=read(base/'frozen.json');prepared=read(base/'preparation.json');state=read(base/'campaign/identity.json')
    text=f'''# Matched AMD audio application baselines — September 20, 2026

All three applications complete the fixed Microsoft ONNX Runtime comparison on
AMD EPYC 9V74, logical CPU 2, .NET 10.0.8 and ORT 1.29.0. Product source is
`1d10d22`, including the qualified WeSpeaker frame precision correction. Exact
source-archived core/Data binaries and the original qualified benchmark consumers
are used. The Windows table remains separate: differing machines and revisions
do not establish a performance change.

{table}

Lower is faster. A Lokad/ORT ratio greater than one means Lokad takes longer.
ASR results are mean complete-corpus totals; pyannote results are per request.
Its three ten-second crops overlap the full thirty-second dialogue. RTF divides
processing time by audio duration. These are descriptive observations from two
fresh timing processes per engine/family, without calibrated confidence bounds
or a parity claim.

Every process performs one full warmup pass and three full measured passes.
Timing order is managed, ORT, ORT, managed for Parakeet, then pyannote, then
Whisper. Six complete conformance processes precede timing. All 792 calls are
retained: 88 conformance, 176 warmup and 528 measured. No outlier or partial
corpus is discarded. Both engines receive the same PCM inputs and FP32 models.
The timer encloses complete feature extraction, neural inference, decoding or
automatic clustering, and output construction. Model loading, file access and
external validation are outside that boundary. Whisper recomputes its pinned
NumPy frontend from PCM inside every native call and pads each clip to thirty
seconds in both engines. Native pyannote combines ORT graphs with pinned
Torch frontend/pooling and NumPy/SciPy/upstream clustering.

ORT uses CPUExecutionProvider, sequential execution, all graph optimizations,
one intra-op/inter-op thread and disabled spinning. Numerical library thread
budgets are one. Every observed worker thread remains on CPU 2. Managed runs
use normal .NET settings without LOKAD, DOTNET or COMPlus overrides.

| Workload | Engine | First process mean seconds | Second process mean seconds |
|---|---|---:|---:|
{chr(10).join(variation)}

Every request passes the original token/text/stop or diarization decision and
centroid checks, input preservation and held-output ownership. Repeated results
within each process agree exactly. Native Whisper's complete Linux feature arrays
are saved for conformance and independently compared with the retained Windows
references at the prospectively fixed absolute 1e-5 bound; every timed call also
records its complete comparison and content hash. This frontend check does not
waive existing full encoder/logit failures. Direct-native WeSpeaker numerical
differences, segmentation-silence differences and wider accuracy limits remain.

Every raw request, timestamp and resource sample is independently audited. The
audit includes {audit['refusals']} additional damaged identity/resource record
refusals, alongside the protocol suite covering real historical records. All
original observed process births are terminal. Full library/model pins verify
before and after execution. The campaign takes {state['seconds']:.3f} wall seconds;
each worker remains within its 3,600-second/14-GiB bounds. Setup times and sampled
peaks below are resource observations, excluded from the API latency table.

| Process | Resource samples | Peak group RSS bytes | Model setup seconds |
|---|---:|---:|---:|
{chr(10).join(resource_rows)}

The remaining ASR rows retain every clip's measured mean:

{header}
{chr(10).join(details)}

Execution payload source: `{prepared['source']}`. Core SHA-256:
`{frozen['files']['bin/Lokad.Onnx.dll']['sha256']}`. Data SHA-256:
`{frozen['files']['bin/Lokad.Onnx.Data.dll']['sha256']}`.
The complete Linux environment, model/source files, actual loaded numerical
libraries, installation/cleanup receipts, process flags and all results are
retained under `artifacts/audio-amd-comparison-v2-20260920`.

[Complete observations](observations-20260920.json) contain every measured,
warmup and conformance result plus per-process/per-pass and per-case totals.
The [Windows Parakeet/pyannote](../comparison/results-20260919.md) and
[Windows Whisper](../whisper-comparison/results-20260919.md) reports remain
unchanged. Product inference remains managed and does not call ORT or Python.
'''
    with report.open('x',encoding='utf-8') as stream:stream.write(text)
    benchmark=ROOT/'BENCHMARK.md';old=benchmark.read_text(encoding='utf-8');anchor='### e5: interleaved independent processes versus native ORT'
    assert old.count(anchor)==1 and '### Audio: matched AMD Microsoft ORT baselines' not in old
    section=f'''### Audio: matched AMD Microsoft ORT baselines

The current `1d10d22` product also has matched complete-application measurements
on **AMD EPYC 9V74, logical CPU 2**, with .NET 10.0.8 and Microsoft ORT **1.29.0**.
The same workloads, complete PCM-to-output boundaries, single-thread settings
and conformance checks are used. This table is separate from the Windows results.

{table}

All 88 conformance, 176 warmup and 528 measured calls pass their application and
ownership checks. Each engine/family has two fresh timing processes, one complete
warmup and three measured passes per process; every call is retained. These are
descriptive results without calibrated confidence or parity claims. The
[complete AMD report](tests/audio/amd-comparison/results-20260920.md) includes
process variation, every ASR clip, memory, actual library identities and the
remaining numerical limitations. Lower is faster; ratios above one favor ORT.

'''
    benchmark.write_text(old.replace(anchor,section+anchor).replace(
        'The audio timing tables remain measurements of `8732831`.',
        'The Windows audio timing tables measure `8732831`; the matched AMD table above measures `1d10d22`.'),encoding='utf-8')
    support=ROOT/'docs/model-support.md';content=support.read_text(encoding='utf-8')
    support.write_text(content.replace('The audio timing tables remain measurements of `8732831`.',
        'The Windows audio timing tables measure `8732831`. The [matched AMD application comparison](../tests/audio/amd-comparison/results-20260920.md) measures `1d10d22` against ORT 1.29.0, retaining every conformance, warmup and measured request.'),encoding='utf-8')
    snapshots=BASE/'closure-tools';snapshots.mkdir()
    for p in folder.glob('*.py'):shutil.copyfile(p,snapshots/p.name)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    reports={p.relative_to(ROOT).as_posix():pin(p) for p in [report,data,Path(__file__),folder/'audit.py']}
    write(BASE/'closed.json',dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,reports=reports,births=receipt['births']))
    print(json.dumps(dict(receipt=pin(BASE/'closed.json'),files=len(files),reports=len(reports))))


if __name__=='__main__':main()
