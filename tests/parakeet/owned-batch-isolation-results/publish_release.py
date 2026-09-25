"""Publish closed relocation evidence and refresh BENCHMARK only for qualified root source."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ORDER = ['current-a','candidate-a','ort-a','ort-b','candidate-b','current-b']
RELEASE_CORE='f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
RELEASE_DATA='a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
CANDIDATE_CORE='e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
CANDIDATE_DATA='01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1'


def pin(path):
    with path.open('rb') as stream: return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def closed(stage):
    suffix = f'{stage}-amd'
    base = ROOT/f'artifacts/parakeet-owned-batch-isolation-{suffix}-20260925'
    if stage == 'root':base=ROOT/'artifacts/parakeet-owned-batch-isolation-root-policy-amd-20260925'
    if stage == 'graphs':base=ROOT/'artifacts/parakeet-owned-batch-graph-qualification-20260925'
    proof = read(base/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(base/name) == wanted, name
    analysis = read(base/'analysis.json'); assert analysis['passed']
    assert proof['files']['analysis.json']==pin(base/'analysis.json')
    return base, proof, analysis


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader(); writer.writerows(rows); return stream.getvalue()


def publish(documents):
    assert all(not (OUT/name).exists() for name in documents), 'Preserve existing publication'
    for name, content in documents.items():
        with (OUT/name).open('x',encoding='utf8',newline='') as stream: stream.write(content)


def observation(base, analysis):
    return json.dumps(dict(closure=pin(base/'closed.json'),**analysis),indent=2,allow_nan=False)+'\n'


def resources(analysis):
    return (f"All owners are terminal. {sum(r['samples'] for r in analysis['resources']):,} resource observations pass; "
        f"peak owned RSS is {max(r['peak_rss'] for r in analysis['resources']):,} bytes.")


def pyannote():
    base, proof, analysis = closed('pyannote-app')
    identities=analysis['identities']
    assert identities['selected']['Lokad.Onnx.dll']['sha256']==RELEASE_CORE
    assert identities['selected']['Lokad.Onnx.Data.dll']['sha256']==RELEASE_DATA
    assert identities['candidate']['Lokad.Onnx.dll']['sha256']==CANDIDATE_CORE
    assert identities['candidate']['Lokad.Onnx.Data.dll']['sha256']==CANDIDATE_DATA
    assert len(analysis['performance']['controls'])==12 and len(analysis['performance']['gates'])==4
    assert proof['admitted']==analysis['performance']['admitted']
    clocks = []; setups = []
    for row in read(base/'collected/identity.json')['runs']:
        if not row['name'].startswith('timing-'): continue
        value = read(base/'collected'/row['name']/'output/result.json')
        setups.append(dict(process=row['name'],seconds=value['setup_seconds']))
        clocks.extend(dict(process=row['name'],**{k:r[k] for k in ['name','pass','phase','start_ticks','end_ticks','frequency','seconds']}) for r in value['records'])
    assert len(setups) == 6 and len(clocks) == 96 and sum(r['phase']=='measured' for r in clocks) == 72
    lines = ['# Dispatch relocation: complete Pyannote comparison and long meetings','',
        '**All application regression gates pass.**' if proof['admitted'] else '**Application regression gates do not all pass.**','',
        '| Fixture | Selected seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT |',
        '|---|---:|---:|---:|---:|']
    for row in analysis['table']:
        lines.append(f"| {row['name']} | {row['selected']['seconds']:.6f} | {row['candidate']['seconds']:.6f} | {row['ort']['seconds']:.6f} | {row['ratios_to_ort']['candidate']:.6f} |")
    lines += ['', 'AMD EPYC 9V74 CPU2, .NET 10.0.8 and ORT 1.29.0 CPUExecutionProvider.',
        'The clock includes frontend, segmentation, embeddings, clustering and owned',
        'public results. Setup and validation are separate. Six fresh processes run',
        'selected, candidate, ORT, ORT, candidate, selected; each has one warmup and',
        'three measured passes per fixture. All clocks and setup intervals remain.','',
        f"Repeatability: {sum(r['passed'] for r in analysis['performance']['controls'])}/12 controls pass; dialogue max/min <=1.10 and crops <=1.20.",
        f"Regression: {sum(r['passed'] for r in analysis['performance']['gates'])}/4 gates pass; candidate/selected <=1.05.",
        'All managed public results match exactly. Native conformance passes the',
        'original limits. Both ten-minute meetings and the thirty-second recovery',
        'preserve native decisions and every selected result, including centroids.','',resources(analysis),'',
        '[All clocks](pyannote-clocks-20260925.csv), [setup](pyannote-setup-20260925.csv),',
        '[complete results and controls](pyannote-observations-20260925.json).','',
        'Closure: `'+pin(base/'closed.json')['sha256']+'`.']
    publish({'pyannote-application-20260925.md':'\n'.join(lines)+'\n',
        'pyannote-clocks-20260925.csv':csv_text(clocks),'pyannote-setup-20260925.csv':csv_text(setups),
        'pyannote-observations-20260925.json':observation(base,analysis)})


def source_qualification(base, analysis):
    adapter=ROOT/'tests/parakeet/owned-batch-isolation-root-policy-amd'
    sys.path.insert(0,str(adapter))
    from source_scope import verify_source,root_files,CORRECTION,FAILED
    from warning_census import compare
    source=verify_source();applied=read(base/'bundle/evidence/root-applied.json')
    assert applied['source_files']==root_files(source)
    assert applied['changed']==sorted(name for name,wanted in applied['source_files'].items() if source['before'].get(name)!=wanted)
    assert len(applied['changed'])==16 and applied['policy_correction']==pin(CORRECTION/'applied.json')
    assert applied['failed_root']==pin(FAILED/'closed.json') and not read(FAILED/'closed.json')['passed']
    assert len(applied['source_files'])==435
    for name,wanted in applied['source_files'].items():assert pin(ROOT/name)==wanted,name
    assert analysis['inventory']['implementation_flags_equal'] and analysis['inventory']['method_bodies_equal']
    assert analysis['inventory']['public_surface_delta_exact'] and analysis['inventory']['data_friend_removed']
    assert analysis['inventory']['public_surface_equal'] is False
    assert (analysis['inventory']['core_methods'],analysis['inventory']['data_methods'])==(3281,697)
    assert analysis['warnings']==compare(base/'collected')
    assert analysis['warnings']['no_new_warning'] and analysis['warnings']['source_warnings']==2
    assert analysis['measured']['Lokad.Onnx.dll']['sha256']==CANDIDATE_CORE
    assert analysis['measured']['Lokad.Onnx.Data.dll']['sha256']==CANDIDATE_DATA
    assert analysis['package']['passed'] and analysis['consumer']['passed']
    for mode,expected in [('suites',(3546,42)),('suite256',(3456,132))]:
        for name,wanted in [('backend',expected),('tensors',(394,0))]:
            actual=analysis[mode][name]
            assert actual['census_exact'] and (actual['passed'],actual['skipped'])==wanted
    return dict(passed=True,source_files=435,exact_measured_method_bodies_and_flags=True,
        policy_metadata_delta_verified=True,policy_guards_unchanged=True,
        corrected_test_arguments_preserved=True,no_new_compiler_warnings=True,warnings=analysis['warnings'])


def root():
    base, _, analysis = closed('root')
    scope=source_qualification(base,analysis)
    assert analysis['root_source_verified']
    expected={'suites':{'backend':(3546,42),'tensors':(394,0)},'suite256':{'backend':(3456,132),'tensors':(394,0)}}
    for mode,suites in expected.items():
        for name,census in suites.items():
            actual=analysis[mode][name]
            assert actual['census_exact'] and (actual['passed'],actual['skipped'])==census
    lines=['# Dispatch relocation: normal root and package qualification','',
        'All 435 root inputs match the measured candidate plus the explicit source-policy correction.',
        'Nine portable hardware guards preserve every original owned-weight test',
        'assertion, helper and case. A normal SDK 10.0.204 build preserves all',
        '3281 Core and 697 Data method bodies and implementation flags.',
        'The only public API change exposes the existing PrepareOwnedMatMulWeights',
        'method; the extra Data friend attribute is removed. All other public and',
        'assembly metadata stays exact. Data now calls the documented public opt-in.',
        'Both original source-policy tests pass unchanged. All 49 test-helper calls',
        'supply the same argument values explicitly, with no optional parameters.',
        'Forty ordinary owned-weight cases pass and the unavailable-hardware case',
        'skips on this FMA-capable VM. The 26 added public tensor cases cover',
        'all positional lengths, exact bits, independent ownership and fallbacks.',
        'Seven public depthwise facts pass, including all 59 observed geometries',
        'against the unchanged generic convolution oracle.',
        'Campaign-specific binary identity fixtures are excluded.','',
        'The first root campaign remains failed at f129f7d8: 392 tensor tests',
        'passed and two source-policy tests failed. The corrected campaign uses',
        'the original sixteen jobs plus restore/build of the existing metadata',
        'inventory helper. Every earlier failure and raw result remains retained.','',
        'The two existing CS8604 warning sites in WideProjectionEntry remain.',
        'Every warning and summary matches the selected release after normalizing',
        'only the campaign directory. There are no new compiler warnings.','',
        '| Full suite | Normal pass / skip | AVX512 disabled pass / skip |',
        '|---|---:|---:|','| Backend | 3,546 / 42 | 3,456 / 132 |','| Tensor | 394 / 0 | 394 / 0 |','',
        'Every existing outcome, 26 slice cases, 41 owned-weight cases and seven',
        'depthwise facts match the',
        'census. Disabling AVX512 skips 93 hardware-specific cases and activates',
        'three unsupported-hardware checks. No additional skip or failure.','',
        'The NuGet package contains the exact built Core and only the existing',
        'Google.Protobuf 3.33.5 dependency. Independent PackageReference consumption',
        'passes model import, matrix and convolution calls, prepared spatial and',
        'Winograd execution, immutable inputs and independently owned outputs.','',
        f"Measured Core: `{analysis['measured']['Lokad.Onnx.dll']['sha256']}`.",
        f"Built Core: `{analysis['built']['Lokad.Onnx.dll']['sha256']}`.",
        f"Built Data: `{analysis['built']['Lokad.Onnx.Data.dll']['sha256']}`.",'',resources(analysis),
        'This qualification provides no new performance measurement.','',
        '[Full census, package and resource evidence](root-observations-20260925.json).','',
        'Closure: `'+pin(base/'closed.json')['sha256']+'`.']
    publish({'root-20260925.md':'\n'.join(lines)+'\n','root-observations-20260925.json':observation(base,dict(**analysis,source_qualification=scope))})


def benchmark():
    root_base, _, root_value = closed('root')
    _, parakeet_proof, parakeet = closed('release-app')
    _, pyannote_proof, pyannote_value = closed('pyannote-app')
    graph_base, graph_proof, graph_value = closed('graphs')
    source_qualification(root_base, root_value)
    assert parakeet_proof['admitted'] and pyannote_proof['admitted']
    assert graph_proof['admitted'] and graph_proof['all_controls_passed']
    current, candidate = parakeet['identities']['current'], root_value['measured']
    assert current['Lokad.Onnx.dll']['sha256'] == RELEASE_CORE
    assert current['Lokad.Onnx.Data.dll']['sha256'] == RELEASE_DATA
    assert parakeet['identities']['candidate'] == candidate
    assert pyannote_value['identities'] == dict(selected=current, candidate=candidate)
    assert graph_value['products'] == dict(current={'Lokad.Onnx.dll': current['Lokad.Onnx.dll']},
                                          candidate={'Lokad.Onnx.dll': candidate['Lokad.Onnx.dll']})
    applied = read(root_base/'bundle/evidence/root-applied.json')
    assert applied['graph_qualification'] == applied['prerequisites']['graphs'] == pin(graph_base/'closed.json')
    assert len(applied['source_files']) == 435
    for name, wanted in applied['source_files'].items():
        assert pin(ROOT/name) == wanted, name
    changed = applied['changed']
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', *changed], cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(['git', 'ls-files', '--', *changed], cwd=ROOT, text=True).splitlines()
    assert set(tracked) == set(changed), 'Commit all qualified product and test additions first'
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%h', '--', *changed], cwd=ROOT, text=True).strip()
    rows = []
    for label, workload, row in [
        ('Parakeet TDT 0.6B V3', 'Transcribe 20 clips / 213.265 seconds of audio', next(row for row in parakeet['table'] if row['is_corpus'])),
        ('Pyannote Community-1', 'Complete diarization of a 30-second dialogue', next(row for row in pyannote_value['table'] if row['audio_seconds'] == 30))]:
        rows.append(f"| {label} | {workload} | {row['candidate']['seconds']:.6f} | {row['ort']['seconds']:.6f} | **{row['ratios_to_ort']['candidate']:.3f}** | Qualified |")
    lookup = {row['key']: row for row in graph_value['performance']}
    for key, label, workload in [
        ('e5-30tok', 'multilingual-e5-small', 'One 30-token forward pass'),
        ('dinov3', 'DINOv3 ViT-S/16', 'One 224x224 image, full weights'),
        ('resnet50', 'ResNet50', 'One 224x224 image, feature export'),
        ('gpt2', 'GPT-2', 'Four-token prefill, empty past state')]:
        row = lookup[key]
        assert row['qualified']
        rows.append(f"| {label} | {workload} | {row['candidate']:.6f} | {row['ort']:.6f} | **{row['ratio']:.3f}** | Qualified |")
    rows += [
        '| DINOv2-small | 224x224 image | — | — | — | Excluded: numerical agreement gate |',
        '| Whisper Large V3 Turbo | Speech transcription | — | — | — | Supported; current-release comparison deferred |']
    for name in ['release-application-20260925.md', 'pyannote-application-20260925.md', 'root-20260925.md']:
        assert (OUT/name).exists(), name
    report = OUT.relative_to(ROOT).as_posix()
    graph_report = 'tests/benchmarks/e5-steady-short-results/qualified-graphs-20260925.md'
    assert (ROOT/graph_report).exists()
    path = ROOT/'BENCHMARK.md'
    previous = path.read_text(encoding='utf8')
    assert 'Core `f95a13c5`' in previous and 'source `dddb60ef`' in previous, 'Preserve unexpected benchmark edits'
    table = '\n'.join(rows)
    document = f'''# CPU benchmarks for the upcoming release

Current repository product, measured on 2026-09-25 UTC. **Lower is better.** Times
are seconds; Lokad / ORT is the latency ratio, so 1.100 means 10.0% more time.

| Model | Measured workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT | Status |
|---|---|---:|---:|---:|---|
{table}

Every numerical timing row uses the same AMD EPYC 9V74 VM, one logical CPU
(CPU 2), .NET 10.0.8 and **Microsoft ONNX Runtime 1.29.0 CPUExecutionProvider**.
Each row is a matched comparison for that workload. Audio rows measure complete
applications; embedding, vision and GPT-2 rows measure prepared graph calls.
The workloads differ, so their absolute times should not be compared to each other.

The selected product is source `{commit}`, measured as Core `{CANDIDATE_CORE[:8]}`
and Data `{CANDIDATE_DATA[:8]}`. Its [normal root and package qualification]({report}/root-20260925.md)
verifies identical computation methods and implementation flags against those
measured binaries, with the documented preparation API exposure and removal of
Data friendship checked separately. Both full test suites pass in normal and AVX512-disabled
modes, and independent NuGet consumption passes. Ordinary mode passes 3,546 backend
and 394 tensor tests; the report records the exact hardware-dependent skips.

## What is timed

Pyannote includes audio frontend, segmentation, speaker embeddings, clustering
and owned diarization results. Its 30-second dialogue uses 21 overlapping windows.
Parakeet includes frontend, encoder/decoder inference, greedy decoding and owned
transcription results. Its total sums the twenty clip means. ORT uses matching
application policies around native inference. Each audio engine has two fresh
timed processes with one warmup and three measured passes per fixture.

Graph timings include a complete forward call returning all owned float arrays:
`Reset`, `Execute` and output materialization for Lokad.Onnx, and `session.run`
for ORT. Inputs are already tensors and batch size is one. Each process uses
1,200 fixed warmups for 30-token e5, 6,000 for 8-token e5 and 600 for the other
cases, followed by 180 measurements. Separate numerical workers run first.
Each comparison includes the previous release: six fresh processes run release,
candidate, ORT, ORT, candidate, release. The table reports the qualified candidate,
which is the current repository product.

Model loading/preparation, file IO, fixture creation, validation and reporting
are outside these timers. ORT uses one intra/inter-op thread, sequential execution
and all graph optimizations. Profilers and managed implementation overrides are
disabled. Every clock is retained and no measurements are trimmed.

## Evidence and coverage

- [Parakeet comparison, all twenty clips and complete clocks]({report}/release-application-20260925.md):
  six measured calls per engine per clip. All 63 repeatability controls, numerical,
  complete public-result, ownership and resource checks pass.
- [Pyannote comparison and complete clocks]({report}/pyannote-application-20260925.md):
  six measured calls per engine for the dialogue. All repeatability, native-result,
  ownership and resource checks pass; both ten-minute meetings and recovery pass.
- [Qualified graph comparisons and complete clocks]({graph_report}):
  e5 at 8, 30, 30 padded to 128, 128 and 512 tokens, DINOv3, ResNet50 and GPT-2.
  Every output is checked against ORT at the unchanged scaled-error bound
  `abs(actual-reference) / max(1, abs(reference)) <= 1e-4`, with exact shapes,
  finite values and ownership checks. A qualified row requires repeated process
  means within 10% for each engine.

DINOv2 is excluded by the [known-divergence registry](tests/Lokad.Onnx.Bench/KnownDivergences.cs):
its registered output exceeds the 1e-4 agreement bound, so the runner withholds
timing. Whisper Large V3 Turbo is supported; a current-release matched timing
refresh and optimization remain deferred. Neither row has a qualified ratio.

Native agreement on these fixtures does not establish general transcription or
diarization accuracy. [Model support and qualification](docs/model-support.md)
describes public APIs, specific exports, accuracy coverage and numerical limits.
The audio APIs live in `Lokad.Onnx.Data`; the core NuGet package contains
`Lokad.Onnx` only.

## Running comparisons

The [graph protocol]({graph_report}),
[Pyannote protocol](tests/parakeet/owned-batch-isolation-pyannote-app-amd/README.md)
and [Parakeet protocol](tests/parakeet/owned-batch-isolation-release-app-amd/README.md)
specify the assets, inputs, process order, boundaries and checks behind the table.
Use the already downloaded `models/multilingual-e5-small/model.onnx` for e5.

For ordinary local comparisons, build and run the repository harness:

```powershell
dotnet build tests/Lokad.Onnx.Bench -c Release --tl:off --nologo -v minimal
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 dinov3 resnet50 gpt2 --mode auto --threads 1 --iters 9
```

The local harness has its own sampling protocol and packaged ORT dependency;
its output is a new measurement. `bench.ps1` additionally includes CLI startup.
Optimization priority is **Parakeet, then Pyannote**, with a matched complete
application target of Lokad / ORT <= 1.05. Whisper optimization is deferred.
'''
    for target in re.findall(r'\]\(([^)]+)\)', document):
        if not target.startswith(('http:', 'https:')):
            assert (ROOT/target).exists(), target
    path.write_text(document, encoding='utf8')


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['pyannote', 'root', 'benchmark']
    globals()[sys.argv[1]]()
