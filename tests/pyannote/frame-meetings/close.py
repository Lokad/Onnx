"""Close the connected replay and publish complete public-output observations."""
from pathlib import Path
import datetime, json, shutil
from prepare import ROOT,BASE,pin,read,write
from vm import ssh


def main():
    audit=read(BASE/'audit.json');assert audit['execution_passed']
    collected=BASE/'collected';receipt=read(collected/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
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
    rows=[]
    for call,comparison in zip(audit['calls'],audit['comparisons'],strict=True):
        rows.append(f"| {call['name']} | {call['seconds']:.3f} | {comparison['maximum_centroid_error']:.9g} | {comparison['mismatch_count']} |")
    scores=[]
    for row in audit['scores']:
        scores.append(f"| {row['name']} | {'Ordinary' if row['timeline']=='intervals' else 'Exclusive'} | {100*row['official']['diarization_error_rate']:.4f}% | {'Yes' if row['unchanged'] else 'No'} |")
    text=f'''# Corrected WeSpeaker frontend: connected meeting replay, September 20, 2026

The source-qualified `1d10d22` Data assembly completes both fixed ten-minute AMI
excerpts and the thirty-second recovery call on AMD EPYC 9V74, CPU 2, .NET 10.0.8.
Public comparisons against the retained Microsoft ORT 1.29 application outputs
{'pass for all three calls' if audit['public_passed'] else 'retain the failures listed in the complete observations'}.
The original native outputs were produced on Windows; no native inference was
repeated for this correctness replay and these durations are not a matched
performance comparison.

| Request | Managed API seconds | Maximum scaled centroid error vs ORT | Public mismatches |
|---|---:|---:|---:|
{chr(10).join(rows)}

The original public checks remain unchanged: discrete decisions must agree,
interval endpoints use an absolute 1e-12 bound and centroids use scaled 1e-4.
All results, inputs and held outputs pass ownership checks. The qualified consumer
binary is reused exactly with the new core/Data assemblies. The correction retains
double precision during WeSpeaker frame preprocessing before the existing FFT;
the model files, segmentation and clustering settings are unchanged.

| Meeting | Timeline | Human DER | Unchanged from ORT |
|---|---|---:|---|
{chr(10).join(scores)}

The official and independent scorers agree on every error component. Aggregate
ordinary DER is {100*audit['aggregates'][0]['diarization_error_rate']:.4f}%; exclusive
DER is {100*audit['aggregates'][1]['diarization_error_rate']:.4f}%. Recovery is excluded.
These results cover the original selected excerpts and scoring policy, not general
diarization accuracy. Complete intermediate tensors are not captured by this
consumer. The separately reported 186 direct-native frontend values above tolerance
and segmentation-silence differences remain recorded limitations.

The one original bounded worker took {audit['process']['seconds']:.3f} wall seconds,
with {audit['process']['samples']:,} resource samples, peak group RSS
{audit['process']['peak_rss']:,} bytes and minimum available RAM
{audit['process']['min_available']:,} bytes. All original supervisor/worker births
are terminal. The audit passes {audit['refusals']['worker']} damaged application
and {audit['refusals']['resource']} damaged resource record refusals. The complete
sample sequence, raw per-call records, input-only verification, source/model pins,
native references, human labels and both score implementations remain in the
artifact. No input, native output or tolerance was changed after inference.

Product source: `{audit['product']['source']}`. Core SHA-256:
`{audit['product']['core']}`. Data SHA-256:
`{audit['product']['data']}`. Consumer SHA-256:
`{audit['product']['runner']}`.

[Complete observations](observations-20260920.json) contain all three outputs,
both comparisons (native and previous managed), every score and resource summary.
See the [original meeting evaluation](../natural-meetings/results-20260920.md)
and [AMD frontend qualification](../frame-product-amd/results-20260920.md).
Artifact: `artifacts/pyannote-frame-meetings-20260920`.
'''
    with report.open('x',encoding='utf-8') as stream:stream.write(text)
    snapshot=BASE/'closure-tools';snapshot.mkdir()
    for path in folder.glob('*.py'):shutil.copyfile(path,snapshot/path.name)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    reports={p.relative_to(ROOT).as_posix():pin(p) for p in [report,data,Path(__file__),folder/'audit_replay.py']}
    write(BASE/'closed.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_passed=True,
        public_passed=audit['public_passed'],accuracy_unchanged=audit['accuracy_unchanged'],files=files,reports=reports,births=receipt['births']))
    print(json.dumps(dict(receipt=pin(BASE/'closed.json'),files=len(files),reports=len(reports),public_passed=audit['public_passed'],accuracy_unchanged=audit['accuracy_unchanged'])))


if __name__=='__main__':main()
