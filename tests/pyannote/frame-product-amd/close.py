"""Close retained source/test/reference evidence without repeating computation."""
from pathlib import Path
import sys, datetime, shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tests/pyannote/filterbank-precision'))
from shared import *
from vm import ssh


def main():
    base=ROOT/'artifacts/wespeaker-frame-product-amd-v4-20260920';audit=read(base/'audit.json')
    reference=ROOT/'artifacts/wespeaker-frame-amd-reference-20260920';spec=read(reference/'manifest.json')
    assert audit['passed'] and audit['source_blobs']==343 and audit['arrays']==742
    assert all(absent(b) for b in [*audit['local_births'],audit['auditor']])
    for name,wanted in audit['bindings'].items():assert pin(ROOT/name)==wanted,name
    verify(spec['files'])
    for path,wanted in spec['numeric'].items():assert pin(path)==wanted
    response=ssh('''import sys,json
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
for identity in %r:
 try:assert psutil.Process(identity['pid']).create_time()!=identity['birth']
 except psutil.NoSuchProcess:pass
print('terminal')
'''%audit['remote_births']);assert response.strip()=='terminal'
    folder=Path(__file__).resolve().parent;data=folder/'observations-20260920.json';report=folder/'results-20260920.md'
    write(data,audit)
    summary=audit['summary'];rows=[]
    for name,label in [('amd-numpy','AMD coefficients / NumPy'),('amd-torch','AMD coefficients / Torch'),
                       ('native-numpy','Native coefficients / NumPy'),('native-torch','Native coefficients / Torch'),
                       ('native-float','Native FP32 frontend'),('windows-product','Windows product')]:
        value=summary[name];rows.append(f"| {label} | {value['failed']:,} | {value['max_scaled']:.9g} |")
    text='''# WeSpeaker frame precision on AMD — September 20, 2026

**The source-archived frontend passes the declared mathematical reference checks
on AMD and all 99 affected tests.** Product `1d10d22` preserves frame preprocessing
in double before the existing FFT. This is frontend qualification; connected
natural-meeting replay with the changed Data assembly remains pending.

All 343 source files match their exact Git blobs. The fresh archive builds on
AMD EPYC 9V74, SDK 10.0.204 / .NET 10.0.8, using CPU 2 and one declared processor.
The test TRX records 99 executed, 99 passed, zero skipped or failed, including
ownership, concurrency, cancellation, invalid inputs, pooling, clustering and CLI
contracts. Two small-signal/DC-offset cases produce exactly identical features
with and without an exactly representable 0.5 offset. Those cases reproduced
262 above-tolerance values on the old Windows DLL.

The public consumer retains all 53 complete arrays: 3,266,560 feature values.
It checks input preservation, dimensions and all held outputs after subsequent
calls. Actual coefficient tables are captured from the loaded Data assembly.
Compared with Windows, AMD differs at four Hamming entries (maximum absolute
5.96046448e-8) and two mel entries (8.79168510e-7). These observed differences
activate the prospectively specified reference branch: both unchanged double
algorithms are run with the actual AMD coefficients and every original input.

Both new references retain all 742 complete stage arrays. NumPy/OpenBLAS and
Torch/MKL agree throughout, maximum scaled error {reference_max:.9g}; independent
scalar preprocessing and direct Fourier checks have maximum {scalar_max:.9g}.
All three duplicate inputs reproduce reference and product arrays exactly.
The reference workers run on Windows CPU 0 with NumPy 2.2.4, Torch 2.11.0+cpu and
one numerical thread. They evaluate the captured AMD function; they do not claim
to reproduce the AMD float instruction sequence.

Every comparison below includes all 3,266,560 final feature values, using
`abs(actual-reference) / max(1, abs(reference)) <= 1e-4`.

| Reference | Failed values | Maximum scaled error |
|---|---:|---:|
{rows}

Only {equal}/53 arrays are bit-identical across Windows and AMD; the complete
cross-host numerical comparison passes. **The 186 direct-native FP32 failures
remain failures.** Qualification uses the prospectively declared independent
frontend reference criterion; no other operator/model bound changes. The
[earlier precision proof](../filterbank-precision/results-20260920.md) explains
the seven incompatible native/reference tolerance intervals. Historical native
failures, silence-segmentation differences and application timing tables remain
unchanged. No new DER or whole-application latency claim follows.

All five final AMD stages exit successfully. Their process groups remain below
the declared 600-second, 2-GiB RSS and available-memory/workspace limits. Every
sample and observed process birth is retained and audited. Both reference workers
also pass their finite bounds; all original processes and the auditor are terminal.
The archive builds report missing Git/SourceLink metadata warnings; the standalone
consumer's affinity API produces a platform-analysis warning. Full logs remain
available rather than claiming warning-free builds.

Preparation and reporting failures are retained. Git archive initially applied
the Windows CRLF policy; two preparations stopped before VM work. An initial
file-mode explanation was wrong, and the preserved diagnosis corrects it. Explicit
per-command LF settings produce exact blobs. The first AMD archive then passes
92 tests and fails seven CLI tests because repository discovery needs the omitted
solution marker; the helper also requires a built CLI. A separate archive includes
both prerequisites and passes all 99 tests. No corpus inference occurred in the
failed attempt. A postprocessing summary helper mismatch is repaired by rereading
saved arrays, with no model or reference rerun.

The final AMD collection retains 752 files; the failed AMD attempt retains 593.
The root VM disk had only 216.8 MB free, so these finite jobs used a new memory-backed
workspace. All bytes have been collected and checked locally; no previous VM model
or evidence was deleted. The canonical checkout is unchanged.

Actual tested core: `{core}`. Actual tested Data: `{data}`.
Product source is `1d10d22`; archive preparation is `e83d2fa`, reference preparation
is `5d9461b`. The core's source is unchanged by this Data-only correction.

[Complete observations](observations-20260920.json) retain every case, reference,
test counter, coefficient identity, process resource summary and loaded assembly.
Artifacts: `artifacts/wespeaker-frame-product-amd-v4-20260920` and
`artifacts/wespeaker-frame-amd-reference-20260920`; earlier v1/v2/v3 preparation
and execution artifacts remain preserved beside them.
'''.format(reference_max=max(r['max_scaled'] for r in audit['stages']),scalar_max=max(r['max_scaled'] for r in audit['scalars']),
           rows='\n'.join(rows),equal=audit['windows_bit_equal_arrays'],core=audit['assemblies']['Lokad.Onnx.dll']['sha256'],data=audit['assemblies']['Lokad.Onnx.Data.dll']['sha256'])
    with report.open('x',encoding='utf-8') as stream:stream.write(text)
    # Copy the current tools into the artifact so further additive tools cannot obscure provenance.
    target=base/'closure-tools';target.mkdir()
    for path in sorted(folder.glob('*.py')):shutil.copyfile(path,target/path.name)
    roots=[ROOT/'artifacts'/name for name in ['wespeaker-frame-product-amd-20260920','wespeaker-frame-product-amd-v2-20260920',
        'wespeaker-frame-product-amd-v3-20260920','wespeaker-frame-product-amd-v4-20260920','wespeaker-frame-amd-reference-20260920']]
    files={rel(p):pin(p) for directory in roots for p in sorted(directory.rglob('*')) if p.is_file()}
    receipt=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=audit['source'],files=files,
                 reports={rel(p):pin(p) for p in [report,data,Path(__file__),folder/'audit.py']},remote_births=audit['remote_births'],
                 local_births=[*audit['local_births'],audit['auditor']],external=spec['files'],numeric=spec['numeric'])
    write(base/'closed.json',receipt);print(json.dumps(dict(receipt=pin(base/'closed.json'),files=len(files),reports=len(receipt['reports']))))


if __name__=='__main__':main()
