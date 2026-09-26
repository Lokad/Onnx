"""One ordinary-runtime, unchanged-product probe of padding compilation history."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
OLD = ROOT / 'tests/parakeet/pad-runtime-diagnostic-amd'
BASE = ROOT / 'artifacts/parakeet-pad-warmup-diagnostic-amd-20260926'
REMOTE_OLD = '/dev/shm/lokad-parakeet-pad-runtime-diagnostic-20260923'
EVIDENCE = ROOT / 'artifacts/parakeet-pad-runtime-diagnostic-amd-20260923'


def previous_closed():
    assert pin(EVIDENCE / 'closed.json')['sha256'] == '2e718d3095eaea8ec79fb263d5ca504564a5e5a83e986ef5535c53505b3864de'
    review = ROOT / 'tests/parakeet/pad-fallback-review-results/observations-20260926.json'
    assert pin(review)['sha256'] == '877d92ec9f2a5c52d4adee3c218b28baec564ea135512648b4f311d6322b5f51'
    value = read(review)
    assert value['passed'] and value['original_screens_remain_rejected'] and not value['admitted']
    for name, wanted in value['inputs'].items():
        assert pin(ROOT / name) == wanted, name


def consumer():
    original = (OLD / 'Screen.cs').read_text(encoding='utf8')
    start = '        var rows = new List<object>(); int index = 0;'
    finish = '        Require(index == 12, "fixed census");'
    body = original[original.index(start):original.index(finish) + len(finish)]
    prime_body = body.replace('events.Begin(index, iteration, marker);',
        'events.PrimeBegin(index, round * 780 + iteration, marker);').replace(
        'events.End(index, iteration, stop);', 'events.PrimeEnd(index, round * 780 + iteration, stop);')
    assert body.count('events.Begin(') == body.count('events.End(') == 1
    prime = '''    static void Prime(JsonDocument censusDocument, string outputDirectory, PadEvents events)
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
        using var output = new FileStream(Path.Combine(outputDirectory, "priming.json"), FileMode.CreateNew);
        JsonSerializer.Serialize(output, new { passed = true, protocol = "pad-census-ten-seconds-after-first-v1",
            pid = Environment.ProcessId, nativeThread = gettid(), frequency = Stopwatch.Frequency,
            began, firstEnd, ended, calls = passes.Count * 9360, passes }, new JsonSerializerOptions { WriteIndented = true });
    }

'''
    result = original.replace('    static void Main(string[] args)', prime + '    static void Main(string[] args)')
    # Only Main's copy receives the call; the generated Prime body is kept exact.
    at = result.index('    static void Main(string[] args)')
    result = result[:at] + result[at:].replace(start,
        '        Prime(censusDocument, outputDirectory, events);\n' + start, 1)
    marker = '    [NonEvent]\n    unsafe void Emit('
    assert result.count(marker) == 1
    result = result.replace(marker, '''    [Event(3, Level = EventLevel.Informational, Keywords = Keywords.Calls)]
    public void PrimeBegin(int fixture, int iteration, long counter) => Emit(3,fixture,iteration,counter);
    [Event(4, Level = EventLevel.Informational, Keywords = Keywords.Calls)]
    public void PrimeEnd(int fixture, int iteration, long counter) => Emit(4,fixture,iteration,counter);
''' + marker)
    assert result[result.index('    static void Main(string[] args)'):].replace(
        '        Prime(censusDocument, outputDirectory, events);\n', '').split('[EventSource')[0] == \
        original[original.index('    static void Main(string[] args)'):].split('[EventSource')[0]
    assert prime_body.replace('events.PrimeBegin(index, round * 780 + iteration, marker);',
        'events.Begin(index, iteration, marker);').replace('events.PrimeEnd(index, round * 780 + iteration, stop);',
        'events.End(index, iteration, stop);') == body
    return result


def prepare():
    assert not BASE.exists()
    previous_closed()
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir()
    originals = {}
    def copy(source, name):
        target = bundle / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    copy(OLD / 'Screen.cs', 'evidence/original-consumer.cs')
    target = bundle / 'source/consumer/Screen.cs'; target.parent.mkdir(parents=True)
    target.write_text(consumer(), encoding='utf8', newline='\n')
    copy(OLD / 'Producer.csproj', 'source/consumer/Producer.csproj')
    copy(ROOT / 'global.json', 'source/global.json')
    for name in ('remote.py', 'remote_prepare.py'):
        copy(OLD / name, 'tools/' + name)
    copy(TOOLS / 'protocol.py', 'tools/protocol.py')
    copy(ROOT / 'tests/parakeet/dispatch-events-amd/remote.py', 'tools/remote_base.py')
    copy(TOOLS / 'README.md', 'README.md')
    old_stage = read(EVIDENCE / 'bundle/stage.json')
    for name in old_stage['files']:
        if name.startswith('evidence/') or name == 'census.json':
            copy(EVIDENCE / 'bundle' / name, name)
    links = {name: dict(source=REMOTE_OLD + '/' + name, identity=value['identity'])
             for name, value in old_stage['links'].items()}
    copy(ROOT / 'tests/parakeet/pad-fallback-review-results/observations-20260926.json', 'evidence/review.json')
    copy(ROOT / 'tests/parakeet/pad-fallback-review-results/report-20260926.md', 'evidence/review.md')
    # Bind the completed owner's evidence as well as its immutable inputs.
    copy(EVIDENCE / 'closed.json', 'evidence/runtime-closed.json')
    copy(EVIDENCE / 'collected/collection.json', 'evidence/runtime-collection.json')
    save(bundle / 'stage.json', dict(passed=True, diagnostic_only=True, links=links,
        products=old_stage['products'], external=old_stage['external'], feed=old_stage['feed'],
        interpreter=old_stage['interpreter'], exporter=old_stage['exporter'],
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for folder in (TOOLS, OLD):
        for p in folder.iterdir():
            if p.is_file():
                if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
                originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals,
        stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), links=len(links))))


if __name__ == '__main__': prepare()
