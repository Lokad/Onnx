"""Publish closed M66 evidence and refresh BENCHMARK only for qualified root source."""
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


def pin(path):
    with path.open('rb') as stream: return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def closed(stage):
    base = ROOT/f'artifacts/parakeet-validated-composition-{stage}-amd-20260924'
    proof = read(base/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(base/name) == wanted, name
    analysis = read(base/'analysis.json'); assert analysis['passed']
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


def graphs():
    base, proof, analysis = closed('graphs')
    assert analysis['clocks'] == 37512 and analysis['measured'] == 8640
    assert analysis['consumer']['branches_locals_exceptions_equal'] and analysis['consumer']['implementation_flags_equal']
    path = ROOT/'tests/parakeet/validated-composition-graphs-amd/statistics.py'
    spec = importlib.util.spec_from_file_location('graph_statistics',path)
    statistics = importlib.util.module_from_spec(spec); spec.loader.exec_module(statistics)
    setups = []
    for row in read(base/'collected/identity.json')['runs']:
        value = read(base/'collected'/row['name']/'output/result.json')
        setups.append(dict(process=row['name'],seconds=value['setup_seconds']))
    assert len(setups) == 72
    lines = ['# Combined candidate: complete graph comparison','',
        '**All graph release gates pass.**' if proof['admitted'] else '**Graph release gates do not all pass.**','',
        '| Case | Selected seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / selected |',
        '|---|---:|---:|---:|---:|---:|']
    for row in analysis['performance']:
        reports = {role:read(base/'collected'/f"timing-{row['key']}-{role}"/'output/result.json') for role in ORDER}
        assert {k:v for k,v in row.items() if k != 'key'} == statistics.summarize(reports)
        lines.append(f"| {row['key']} | {row['current']:.6f} | {row['candidate']:.6f} | {row['ort']:.6f} | {row['ratio']:.6f} | {row['candidate_over_current']:.6f} |")
    lines += ['', 'AMD EPYC 9V74 CPU2, .NET 10.0.8 and ORT 1.29.0 CPUExecutionProvider.',
        'Lokad times Reset, Execute and owned output arrays; ORT times session.run.',
        'Setup, input creation and validation are separate. Native inference uses',
        'one intra/inter-op thread, sequential execution and all graph optimizations.','',
        'The 24 numerical processes run first, three calls each. Each of eight cases',
        'then uses six fresh timing processes: selected, candidate, ORT, ORT, candidate,',
        'selected. Every process retains 600 warmups and 180 measured calls.',
        'All 37,512 clocks, 8,640 measurements and 72 setup intervals are retained;',
        'exact clock fractions give equal process weights. No sample is trimmed.','',
        f"Repeatability: {sum(c['passed'] for r in analysis['performance'] for c in r['controls'])}/24 controls pass (process max/min <=1.10).",
        f"Regression: {sum(r['regression_passed'] for r in analysis['performance'])}/8 gates pass (candidate/selected <=1.05).",'',
        'Every candidate output matches selected bytes. All products meet fresh',
        'ORT scaled error <=1e-4, exact shapes, finiteness, unchanged inputs and',
        'held-output ownership checks. The previously qualified warmed consumer',
        'is reused unchanged; no profiler or build overlaps the comparison.','',
        resources(analysis),'',
        '[All clocks](graphs-clocks-20260924.csv), [all setups](graphs-setup-20260924.csv),',
        '[complete controls, consumer and resource evidence](graphs-observations-20260924.json).','',
        'Closure: `'+pin(base/'closed.json')['sha256']+'`.']
    publish({'graphs-20260924.md':'\n'.join(lines)+'\n',
        'graphs-clocks-20260924.csv':(base/'clocks.csv').read_text(),
        'graphs-setup-20260924.csv':csv_text(setups),
        'graphs-observations-20260924.json':observation(base,analysis)})


def pyannote():
    base, proof, analysis = closed('pyannote-app')
    clocks = []; setups = []
    for row in read(base/'collected/identity.json')['runs']:
        if not row['name'].startswith('timing-'): continue
        value = read(base/'collected'/row['name']/'output/result.json')
        setups.append(dict(process=row['name'],seconds=value['setup_seconds']))
        clocks.extend(dict(process=row['name'],**{k:r[k] for k in ['name','pass','phase','start_ticks','end_ticks','frequency','seconds']}) for r in value['records'])
    assert len(setups) == 6 and len(clocks) == 96 and sum(r['phase']=='measured' for r in clocks) == 72
    lines = ['# Combined candidate: complete Pyannote comparison and long meetings','',
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
        '[All clocks](pyannote-clocks-20260924.csv), [setup](pyannote-setup-20260924.csv),',
        '[complete results and controls](pyannote-observations-20260924.json).','',
        'Closure: `'+pin(base/'closed.json')['sha256']+'`.']
    publish({'pyannote-application-20260924.md':'\n'.join(lines)+'\n',
        'pyannote-clocks-20260924.csv':csv_text(clocks),'pyannote-setup-20260924.csv':csv_text(setups),
        'pyannote-observations-20260924.json':observation(base,analysis)})


def root():
    base, _, analysis = closed('root')
    assert analysis['root_source_verified'] and analysis['inventory']['implementation_flags_equal']
    assert (analysis['inventory']['core_methods'],analysis['inventory']['data_methods']) == (3251,697)
    expected = {'suites':{'backend':(3471,41),'tensors':(368,0)},'suite256':{'backend':(3381,131),'tensors':(368,0)}}
    for mode, suites in expected.items():
        for name, census in suites.items():
            actual = analysis[mode][name]
            assert actual['census_exact'] and (actual['passed'],actual['skipped']) == census
    source = read(base/'bundle/evidence/root-applied.json')['source_files']; assert len(source) == 425
    for name, wanted in source.items(): assert pin(ROOT/name) == wanted, name
    lines = ['# Combined candidate: normal root and package qualification','',
        'All 425 root inputs match the admitted source. A normal SDK 10.0.204 build',
        'preserves all 3,251 Core and 697 Data method bodies, implementation flags',
        'and public interfaces against the measured candidate.','',
        '| Full suite | Normal pass / skip | AVX512 disabled pass / skip |',
        '|---|---:|---:|','| Backend | 3,471 / 41 | 3,381 / 131 |','| Tensor | 368 / 0 | 368 / 0 |','',
        'Every original test outcome and all 22 recurrence and 25 copying cases',
        'match the prospective census. Disabling AVX512 skips 93 hardware-specific',
        'cases and activates three unsupported-hardware checks. No new skip or failure.','',
        'The NuGet package contains the exact built Core and only the existing',
        'Google.Protobuf 3.33.5 dependency. Independent PackageReference consumption',
        'passes model import, matrix and convolution calls, prepared spatial and',
        'Winograd execution, immutable inputs and independently owned outputs.','',
        f"Measured Core: `{analysis['measured']['Lokad.Onnx.dll']['sha256']}`.",
        f"Built Core: `{analysis['built']['Lokad.Onnx.dll']['sha256']}`.",
        f"Built Data: `{analysis['built']['Lokad.Onnx.Data.dll']['sha256']}`.",'',resources(analysis),
        'This qualification provides no new performance measurement.','',
        '[Full census, package and resource evidence](root-observations-20260924.json).','',
        'Closure: `'+pin(base/'closed.json')['sha256']+'`.']
    publish({'root-20260924.md':'\n'.join(lines)+'\n','root-observations-20260924.json':observation(base,analysis)})


def benchmark():
    import qualified_graphs
    rb, _, ra = closed('root'); _, pp, pa = closed('app')
    _, yp, ya = closed('pyannote-app'); ga = qualified_graphs.admission()
    assert pp['admitted'] and yp['admitted'] and ga['admitted'] and ra['root_source_verified']
    assert pa['identities']['candidate'] == ya['identities']['candidate'] == ra['measured']
    assert ga['products']['candidate']['Lokad.Onnx.dll'] == ra['measured']['Lokad.Onnx.dll']
    applied = read(rb/'bundle/evidence/root-applied.json')
    assert applied['graph_qualification'] == pin(qualified_graphs.BASE/'closed.json')
    assert ga['source_closures'] == dict(original_graphs=applied['prerequisites']['graphs'],e5_successor=applied['prerequisites']['e5'])
    source = applied['source_files']; assert len(source) == 425
    for name, wanted in source.items(): assert pin(ROOT/name) == wanted, name
    changed = read(rb/'bundle/evidence/root-applied.json')['changed']
    assert not subprocess.check_output(['git','diff','HEAD','--name-only','--',*changed],cwd=ROOT,text=True).strip()
    commit = subprocess.check_output(['git','log','-1','--format=%h','--',*changed],cwd=ROOT,text=True).strip()
    rows = []
    for label, work, row in [('Parakeet TDT 0.6B V3','Transcribe 20 clips / 213.265 seconds of audio',next(r for r in pa['table'] if r['is_corpus'])),
        ('Pyannote Community-1','Complete diarization of a 30-second dialogue',next(r for r in ya['table'] if r['audio_seconds']==30))]:
        rows.append(f"| {label} | {work} | {row['candidate']['seconds']:.6f} | {row['ort']['seconds']:.6f} | **{row['ratios_to_ort']['candidate']:.3f}** | Qualified |")
    lookup = {r['key']:r for r in ga['performance']}
    for key, label, work in [('e5-30tok','multilingual-e5-small','One 30-token forward pass'),('dinov3','DINOv3 ViT-S/16','One 224x224 image, full weights'),
        ('resnet50','ResNet50','One 224x224 image, feature export'),('gpt2','GPT-2','Four-token prefill, empty past state')]:
        row = lookup[key]; assert row['qualified']
        rows.append(f"| {label} | {work} | {row['candidate']:.6f} | {row['ort']:.6f} | **{row['ratio']:.3f}** | Qualified |")
    rows += ['| DINOv2-small | 224x224 image | — | — | — | Excluded: numerical agreement gate |',
        '| Whisper Large V3 Turbo | Speech transcription | — | — | — | Supported; current-release comparison deferred |']
    for name in ['application-20260924.md','qualified-graphs-20260924.md','pyannote-application-20260924.md','root-20260924.md']:
        assert (OUT/name).exists(), name
    path = ROOT/'BENCHMARK.md'; document = path.read_text(encoding='utf8')
    start = document.index('| Parakeet TDT'); end = document.index('\n\nEvery numerical timing row',start)
    document = document[:start]+'\n'.join(rows)+document[end:]
    document, count = re.subn(r'measured on \d{4}-\d{2}-\d{2} UTC', 'measured on 2026-09-24 UTC', document); assert count == 1
    document = document.replace('so 1.154 means 15.4% more time','so 1.100 means 10.0% more time')
    before = 'uses 600 fixed warmups and 180 measurements. Separate numerical workers run first.'
    assert document.count(before) == 1
    document = document.replace(before,
        'uses 1,200 fixed warmups for 30-token e5 and 600 for the other graph cases,\n'
        'followed by 180 measurements. Separate numerical workers run first.')
    start = document.index('The selected product is source'); end = document.index('\n## What is timed',start)
    document = document[:start]+f'''The selected product is source `{commit}`, measured as Core `{ra['measured']['Lokad.Onnx.dll']['sha256'][:8]}` and
Data `{ra['measured']['Lokad.Onnx.Data.dll']['sha256'][:8]}`. The [normal root and package qualification](tests/parakeet/validated-composition-results/root-20260924.md)
verifies all compiled methods, implementation flags and public interfaces, full
backend/tensor suites in both instruction modes and independent NuGet consumption.
Ordinary mode passes 3,471 backend and 368 tensor tests; the hardware-dependent
skip census in each mode is recorded in that report.
''' + document[end:]
    for before, after in [('wide-entry-first-use-results/pyannote-20260923.md','validated-composition-results/pyannote-application-20260924.md'),
        ('wide-entry-first-use-results/application-20260923.md','validated-composition-results/application-20260924.md'),
        ('wide-entry-first-use-results/graphs-20260923.md','validated-composition-results/qualified-graphs-20260924.md'),
        ('wide-entry-first-use-graphs-amd/README.md','validated-composition-results/qualified-graphs-20260924.md'),
        ('wide-entry-first-use-pyannote-app-amd/README.md','validated-composition-pyannote-app-amd/README.md'),
        ('wide-entry-first-use-app-amd-v2/README.md','validated-composition-app-amd/README.md')]:
        assert document.count(before) == 1; document = document.replace(before,after)
    assert 'wide-entry-first-use' not in document and '20260923' not in document
    for target in re.findall(r'\]\(([^)]+)\)',document):
        if not target.startswith(('http:','https:')): assert (ROOT/target).exists(), target
    path.write_text(document,encoding='utf8')


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['graphs','pyannote','root','benchmark']
    globals()[sys.argv[1]]()
