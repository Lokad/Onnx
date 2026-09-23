"""Publish current-release comparisons and replace the benchmark landing page."""
from pathlib import Path
import json,shutil,sys
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/benchmarks/release-amd-v2'))
from protocol import pin,read,save,CASES,ORDER
from statistics import summarize
BASE=ROOT/'artifacts/release-graph-baseline-amd-v2-20260923'

def main():
    target=OUT/'results-20260923.md';assert not target.exists()
    proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    a=read(BASE/'analysis.json');assert a['passed']
    for row in a['performance']:
        assert {k:v for k,v in row.items() if k!='key'}==summarize({role:read(BASE/'collected'/('timing-'+row['key']+'-'+role)/'output/result.json') for role in ORDER})
    shutil.copyfile(BASE/'clocks.csv',OUT/'clocks-20260923.csv')
    save(OUT/'observations-20260923.json',dict(closure=pin(BASE/'closed.json'),payload=pin(BASE/'payload.json'),analysis=a))
    setup=[]
    for name in read(BASE/'collected/identity.json')['runs']:
        if name['name'].startswith(('verify-','timing-')):
            v=read(BASE/'collected'/name['name']/'output/result.json');setup.append(dict(process=name['name'],seconds=v['setup_seconds']))
    save(OUT/'setup-20260923.json',setup)
    lines=['# Current release graph comparisons versus Microsoft ORT','',
        'Actual selected Core `521bae17`, AMD EPYC 9V74 CPU2, .NET 10.0.8 and ORT 1.29.0.',
        'All values time complete graph calls returning owned float arrays. All means',
        'retain every fixed measured call; setup and all warmups are separate.','',
        '| Workload | Lokad.Onnx ms | Microsoft ORT ms | Lokad / ORT | Repeatability |',
        '|---|---:|---:|---:|---|']
    for r in a['performance']:
        lines.append(f"| {r['key']} | {r['current']*1000:.6f} | {r['ort']*1000:.6f} | {r['ratio']:.6f} | {'PASS' if r['qualified'] else 'FAIL — ratio unqualified'} |")
    lines+=['','Each of eight workloads has two separate three-call verification workers',
        'before any timing worker. Four fresh timing processes run current,ORT,ORT,current;',
        'each retains 60 warmups and 60 measured calls. There are 3,888 total clocks, including',
        '48 verification calls and 1,920 measured calls. No calibration, trimming or retry.',
        'Managed timers include Reset, Execute and materialization of all output arrays;',
        'native timers include session.run and all returned arrays. Input creation,',
        'load/preparation, output validation, hashing and reporting are outside the timer.',
        'Both runtimes use ordinary defaults without profiling or implementation overrides.',
        'ORT uses CPUExecutionProvider, sequential execution, all graph optimizations,',
        'and one intra/inter-op thread. Product methods match the qualified root release.','',
        'Every call passes unchanged native reference bounds, finiteness, output shape/name,',
        'determinism, read-only input and retained output checks. Collected full outputs',
        'also pass independent fresh-ORT comparisons at abs(actual-reference)/max(1,abs(reference))',
        '<=1e-4. Each role/case requires repeated process means max/min<=1.10.','',
        '| Case | Current repeat ratio | ORT repeat ratio |', '|---|---:|---:|']
    for r in a['performance']:lines.append(f"| {r['key']} | {r['controls'][0]['ratio']:.6f} | {r['controls'][1]['ratio']:.6f} |")
    lines+=['',f"All {sum(r['samples'] for r in a['resources'])} resource observations pass; peak owned RSS is {max(r['peak_rss'] for r in a['resources']):,} bytes.",
        'Each recorded process and thread is confined to CPU2; the supervisor uses CPU0.',
        'All recorded owner identities are terminal. The original consumer build stopped',
        'at two ITensor shape-property errors before numerical or timing work. The v2',
        'correction changes those checks to Dims, preserving all boundaries and gates.',
        'During verification preflight, the supervisor was briefly paused with no child',
        'active to deduplicate 918 immutable closed files into 38 content groups. Every path',
        'and hash was preserved; 737,722,708 duplicate bytes were reclaimed. Independent',
        'local restoration copies and complete payload verification precede resumption.',
        'No measured worker overlapped maintenance. The 12 GiB launch bound was unchanged.','',
        '[Frozen protocol](../release-amd-v2/README.md), [all raw clocks](clocks-20260923.csv),',
        '[all setup intervals](setup-20260923.json), [complete controls and resources](observations-20260923.json).','',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.',
        'Core: `521bae1702849ca23dda586515e7cbabaac2d1eabdff04dc90a7ba76059e93fb`.','']
    target.write_text('\n'.join(lines),encoding='utf8')
    py=ROOT/'artifacts/pyannote-winograd-product-app-amd-20260923';pa=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
    assert pin(py/'closed.json')['sha256']=='dc2c7b9f5086ab9b4ee615b9dad7643eaf4c1ed65c1cc71b3e76ee794237d88e'
    assert pin(pa/'closed.json')['sha256']=='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3'
    py_analysis=read(py/'analysis.json');pa_analysis=read(pa/'analysis.json')
    assert py_analysis['performance']['admitted'] and pa_analysis['performance']['baseline_valid']
    py_row=next(r for r in py_analysis['table'] if r['name']=='dialogue-30s')
    pa_row=next(r for r in pa_analysis['table'] if r['is_corpus'])
    lookup={r['key']:r for r in a['performance']}
    rows=[]
    for label,work,row,role in [('Pyannote Community-1','Complete diarization of a 30-second dialogue',py_row,'candidate'),('Parakeet TDT 0.6B V3','Transcribe 20 clips / 213.265 seconds of audio',pa_row,'current')]:
        rows.append(f"| {label} | {work} | {row[role]['seconds']:.6f} | {row['ort']['seconds']:.6f} | **{row['ratios_to_ort'][role]:.3f}** | Qualified |")
    for key,label,work in [('e5-30tok','multilingual-e5-small','One 30-token forward pass'),('dinov3','DINOv3 ViT-S/16','One 224x224 image, full weights'),('resnet50','ResNet50','One 224x224 image, feature export'),('gpt2','GPT-2','Four-token prefill, empty past state')]:
        r=lookup[key]
        ratio=f"**{r['ratio']:.3f}**" if r['qualified'] else '—'
        rows.append(f"| {label} | {work} | {r['current']:.6f} | {r['ort']:.6f} | {ratio} | {'Qualified' if r['qualified'] else 'Repeatability failed; ratio withheld'} |")
    rows+=['| DINOv2-small | 224x224 image | — | — | — | Excluded: numerical agreement gate |',
        '| Whisper Large V3 Turbo | Speech transcription | — | — | — | Supported; current-release comparison deferred |']
    # Keep this landing page limited to release state and reproducible current evidence.
    document='''# CPU benchmarks for the upcoming release

Current repository product, measured on 2026-09-23. **Lower is better.** Times
are seconds; Lokad / ORT is the latency ratio, so 1.156 means 15.6% more time.

| Model | Measured workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT | Status |
|---|---|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

Every numerical timing row uses the same AMD EPYC 9V74 VM, one logical CPU
(CPU2), .NET 10.0.8 and **Microsoft ONNX Runtime 1.29.0 CPUExecutionProvider**.
Each row is a matched comparison for that workload. Audio rows measure complete
applications; the embedding, vision and GPT-2 rows measure prepared graph calls.
The workloads differ, so their absolute times should not be compared to each other.

The selected product is source `94a550de`, measured as Core `521bae17` and
Data `f3b9aa81`. The [normal root and package qualification](tests/pyannote/winograd-product-results/root-20260923.md)
verifies the same compiled methods and public interfaces, 3,449 backend tests,
343 tensor tests, both instruction widths and independent NuGet consumption.
The 41 existing AMD test skips are recorded in that report.

## What is timed

Pyannote includes audio frontend, segmentation, speaker embeddings, clustering
and owned diarization results. Its 30-second dialogue uses 21 overlapping windows.
Parakeet includes frontend, encoder/decoder inference, greedy decoding and owned
transcription results. Its total sums the twenty clip means. The ORT comparisons
use matching application policies around native inference. Each audio engine has
two fresh timed processes with one warmup and three measured passes per fixture.

Graph timings include a complete forward call returning all owned float arrays:
`Reset`, `Execute` and output materialization for Lokad.Onnx, and `session.run`
for ORT. Inputs are already tensors, batch size is one, and each timed process
uses 60 fixed warmups and 60 measurements. Separate numerical workers run first.
Each case uses four fresh processes in Lokad,ORT,ORT,Lokad order.

Model loading/preparation, file IO, fixture creation, validation and reporting
are outside these timers. ORT uses one intra/inter-op thread, sequential execution
and all graph optimizations. Profilers and managed implementation overrides are
disabled. All clocks are retained; no measurements are trimmed or retried.

## Evidence and coverage

- [Pyannote comparison and complete clocks](tests/pyannote/winograd-product-results/application-20260923.md):
  six measured calls per engine for the dialogue. All repeatability, native-result,
  ownership and resource checks pass; both ten-minute meetings and recovery pass.
- [Parakeet comparison, all twenty clips and complete clocks](tests/parakeet/winograd-baseline-amd/results-20260923.md):
  six measured calls per engine per clip. All 42 repeatability controls, native/public
  result checks and resource checks pass.
- [Current graph comparisons and complete clocks](tests/benchmarks/release-results/results-20260923.md):
  includes e5 at 8, 30, 30 padded to 128, 128 and 512 tokens, DINOv3, ResNet50 and GPT-2.
  Every output is checked against fresh ORT at the unchanged scaled-error bound
  `abs(actual-reference) / max(1, abs(reference)) <= 1e-4`, with exact shapes,
  finite values and ownership checks. A qualified row requires repeated process
  means within 10% for each engine.

DINOv2 remains excluded by the [known-divergence registry](tests/Lokad.Onnx.Bench/KnownDivergences.cs):
its registered output exceeds the 1e-4 agreement bound, so the runner withholds
timing. Whisper Large V3 Turbo is supported, but a current-release matched timing
refresh and further optimization are deferred. Neither row has a qualified
current-release ratio.

Native agreement on these fixtures does not establish general transcription or
diarization accuracy. [Model support and qualification](docs/model-support.md)
describes the public APIs, specific exports, accuracy coverage and remaining
numerical limitations. The audio APIs live in `Lokad.Onnx.Data `; the core NuGet
package contains `Lokad.Onnx` only.

## Running comparisons

The [frozen graph protocol](tests/benchmarks/release-amd-v2/README.md),
[Pyannote protocol](tests/pyannote/winograd-product-app-amd/README.md) and
[Parakeet protocol](tests/parakeet/winograd-baseline-amd/README.md) specify the
assets, inputs, process order, boundaries and checks behind these tables.
Use the already downloaded `models/multilingual-e5-small/model.onnx` for e5.

For ordinary local model comparisons, build and run the repository harness:

```powershell
dotnet build tests/Lokad.Onnx.Bench -c Release --tl:off --nologo -v minimal
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 dinov3 resnet50 gpt2 --mode auto --threads 1 --iters 9
```

The local harness has its own sampling protocol and packaged ORT dependency;
its output is a new measurement. `bench.ps1` additionally includes CLI startup.
Current optimization priority is **Pyannote, then Parakeet**, with a matched
application latency target of Lokad / ORT <= 1.05. Whisper optimization is deferred.
'''
    (ROOT/'BENCHMARK.md').write_text(document,encoding='utf8')
    print(json.dumps(dict(report=pin(target),benchmark=pin(ROOT/'BENCHMARK.md'))))

if __name__=='__main__':main()
