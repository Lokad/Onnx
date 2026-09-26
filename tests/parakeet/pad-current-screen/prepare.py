"""One fixed-warmup ABBA screen of the current-root padding composition."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from census import census
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
OLD = ROOT / 'tests/parakeet/pad-dispatch-screen'
BASE = ROOT / 'artifacts/parakeet-pad-current-screen-amd-20260926'
SOURCE = ROOT / 'artifacts/parakeet-pad-current-source-20260926'
BUILD = ROOT / 'artifacts/parakeet-pad-current-build-amd-20260926'
CURRENT = ROOT / 'artifacts/parakeet-owned-batch-isolation-root-policy-amd-20260925'


def previous_closed():
    for folder, digest in [(BUILD, '1835c79eda505c056cf796702ca734b6e43c65e066b3e54bb566ac25885c4018'),
            (CURRENT, 'c77ef606c76508144c5656bdbe48aeac8948ce28396c8bee645587d31609c475')]:
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder / name) == wanted, name
    assert pin(SOURCE / 'prepared.json')['sha256'] == 'f9eb6c4a26362529d4fe19089e35201d5e26f318561445b58078d05c1df5dbea'
    for name, wanted in read(SOURCE / 'prepared.json')['before'].items(): assert pin(ROOT / name) == wanted, name
    analysis = read(BUILD / 'analysis.json')
    assert analysis['source_prepared'] == pin(SOURCE / 'prepared.json')
    assert analysis['measured'] == read(CURRENT / 'analysis.json')['built']
    assert analysis['inventory']['original_padcore_exact'] and analysis['inventory']['helper_matches_reviewed_original']
    assert [analysis['suites'][name]['passed'] for name in ('pad-tests', 'pad-tests-256')] == [6, 6]
    for name in ('census.py', 'score.py', 'test_score.py', 'Prototype.csproj'):
        assert pin(TOOLS / name) == pin(OLD / name), name


def consumer():
    original = (OLD / 'Screen.cs').read_text(encoding='utf8')
    start = '        var rows = new List<object>(); int index = 0;'
    end = '        Require(index == 12, "fixed census");'
    body = original[original.index(start):original.index(end) + len(end)]
    prime_body = body.replace('long ticks = Stopwatch.GetTimestamp() - start;',
        'long stop = Stopwatch.GetTimestamp();\n                long ticks = stop - start;').replace(
        'clocks.Add(new { iteration, warmup = iteration < 600, ticks });',
        'clocks.Add(new { iteration, warmup = iteration < 600, start, stop, ticks });')
    assert prime_body != body
    method = '''    static long Prime(JsonDocument censusDocument, string outputDirectory)
    {
        long began = Stopwatch.GetTimestamp(), firstEnd = 0, ended = 0;
        var passes = new List<object>();
        for (int round = 0; ; round++)
        {
            Require(round < 16 && Stopwatch.GetElapsedTime(began).TotalSeconds < 180, "priming bound");
            long roundStart = Stopwatch.GetTimestamp();
''' + prime_body + '''
            ended = Stopwatch.GetTimestamp();
            passes.Add(new { round, start = roundStart, end = ended, rows });
            if (round == 0) firstEnd = ended;
            if ((ended - firstEnd) / (double)Stopwatch.Frequency >= 10) break;
        }
        Require(Stopwatch.GetElapsedTime(began).TotalSeconds < 180, "priming duration");
        using (var output = new FileStream(Path.Combine(outputDirectory, "priming.json"), FileMode.CreateNew))
            JsonSerializer.Serialize(output, new { passed = true, protocol = "pad-census-ten-seconds-after-first-v1",
                pid = Environment.ProcessId, frequency = Stopwatch.Frequency, began, firstEnd, ended,
                calls = passes.Count * 9360, passes }, new JsonSerializerOptions { WriteIndented = true });
        return Stopwatch.GetTimestamp();
    }

'''
    invocation = '        long suffixStart = Prime(censusDocument, Path.GetDirectoryName(Path.GetFullPath(args[3]))!);\n'
    result = original.replace('    static void Main(string[] args)', method + '    static void Main(string[] args)')
    at = result.index('    static void Main(string[] args)')
    result = result[:at] + result[at:].replace(start, invocation + start, 1).replace(
        '            calls = 9360,', '            suffixStart, calls = 9360,', 1)
    assert result[result.index('    static void Main(string[] args)'):].replace(invocation, '').replace(
        '            suffixStart, calls = 9360,', '            calls = 9360,') == original[original.index('    static void Main(string[] args)'):]
    assert prime_body.replace('long stop = Stopwatch.GetTimestamp();\n                long ticks = stop - start;',
        'long ticks = Stopwatch.GetTimestamp() - start;').replace(
        'clocks.Add(new { iteration, warmup = iteration < 600, start, stop, ticks });',
        'clocks.Add(new { iteration, warmup = iteration < 600, ticks });') == body
    return result


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    target = bundle / 'source/consumer/Screen.cs'; target.parent.mkdir(parents=True)
    target.write_text(consumer(), encoding='utf8', newline='\n')
    copy(OLD / 'Screen.cs', bundle / 'evidence/original-consumer.cs')
    copy(TOOLS / 'Prototype.csproj', bundle / 'source/consumer/Prototype.csproj')
    copy(ROOT / 'global.json', bundle / 'source/global.json')
    for name in ('protocol.py', 'remote.py', 'remote_prepare.py'):
        copy(TOOLS / name, bundle / 'tools' / name)
    copy(TOOLS / 'README.md', bundle / 'README.md')
    shutil.copy2(ROOT / 'PLAN.md', bundle / 'prospective-plan.md')
    products = {}
    for role, folder in [('current', CURRENT / 'collected/runtime'), ('candidate', BUILD / 'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file(): copy(p, bundle / 'runtimes' / role / p.name)
        products[role] = {name: pin(bundle / 'runtimes' / role / name) for name in ('Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll')}
    assert products['current'] == read(CURRENT / 'analysis.json')['built']
    assert products['candidate'] == read(BUILD / 'analysis.json')['built']
    for name in ('closed.json', 'payload.json', 'analysis.json'):
        copy(BUILD / name, bundle / 'evidence' / ('build-' + name))
    copy(BUILD / 'collected/collection.json', bundle / 'evidence/build-collection.json')
    copy(ROOT / 'tests/parakeet/pad-warmup-diagnostic-results/observations-20260926.json', bundle / 'evidence/warmup-diagnostic.json')
    save(bundle / 'census.json', census())
    save(bundle / 'stage.json', dict(passed=True, products=products, root_product_changed=False,
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals,
        stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), products=products)))


if __name__ == '__main__': prepare()
