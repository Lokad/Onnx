"""Correct only the synthetic stimulus; preserve the failed run and product DLLs."""
import difflib
import shutil
from common import *
import qualify

TARGET = ROOT / 'artifacts/pyannote-request-contexts-v2-20260921'


def main():
    assert not TARGET.exists()
    receipt = BASE / 'failure-closed.json'
    assert pin(receipt)['sha256'] == '6dcd75a7bf86f91bec262339253a132eb6a426d4277f9041f19b87addb40d71d'
    failure = read(receipt); assert not failure['passed']; verify(failure['files'])
    for identity in failure['identities']: terminal(identity)
    TARGET.mkdir(); (TARGET / 'logs').mkdir()
    shutil.copytree(BASE / 'source', TARGET / 'source', ignore=shutil.ignore_patterns('bin', 'obj'))
    shutil.copytree(BASE / 'runtime', TARGET / 'runtime')
    for name in ['prepared.json', 'builds.json', 'instructions.json', 'candidate.patch', 'source-changes.json']:
        shutil.copy2(BASE / name, TARGET / name)
    for row in read(BASE / 'builds.json')['runs']:
        for suffix in ['.log', '.samples.jsonl']:
            shutil.copy2(BASE / 'logs' / (row['name']+suffix), TARGET / 'logs' / (row['name']+suffix))
    path = TARGET / 'source/tests/Lokad.Onnx.Backend.Tests/PipelineRequestTests.cs'
    before = path.read_text(encoding='utf8')
    replacements = [
        ('var a = Pcm(.1f); var b = Pcm(.3f);', 'var a = new float[160000]; var b = Pcm(.3f);'),
        ('Math.PI * 437 * i / 16000', 'Math.PI * (i < 80000 ? 437 : 997) * i / 16000'),
        ('            var mean = new NodeProto { OpType = "ReduceMean" }; mean.Input.Add("fbank_features");',
         '            // Squared centered features preserve variation; their signed mean intentionally vanishes.\n'
         '            var square = new NodeProto { OpType = "Mul" }; square.Input.Add(new[] { "fbank_features", "fbank_features" }); square.Output.Add("squared"); encoder.Graph.Node.Add(square);\n'
         '            var mean = new NodeProto { OpType = "ReduceMean" }; mean.Input.Add("squared");')]
    after = before
    for old, new in replacements:
        assert after.count(old) == 1, old
        after = after.replace(old, new)
    path.write_text(after, encoding='utf8')
    source_files = {}
    for previous in (BASE / 'source').rglob('*'):
        name = previous.relative_to(BASE / 'source')
        if not previous.is_file() or {'bin', 'obj'}.intersection(name.parts): continue
        changed = TARGET / 'source' / name
        if changed != path: assert pin(previous) == pin(changed), str(name)
        source_files[rel(changed)] = pin(changed)
    for previous in (BASE / 'runtime').iterdir():
        if previous.is_file(): assert pin(previous) == pin(TARGET / 'runtime' / previous.name)
    save(TARGET / 'successor-prepared.json', dict(passed=True, failure=pin(receipt), source_files=source_files,
        tool=pin(Path(__file__)), reused_preparation=pin(BASE / 'prepared.json'),
        runtime={p.name:pin(p) for p in (TARGET / 'runtime').glob('*.dll')},
        test_change=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True))),
        scope='Only synthetic test stimulus changes: silence versus two-tone PCM; mean squared centered features. Product code/binaries and assertions are unchanged.'))
    qualify.BASE = TARGET; qualify.monitor.BASE = TARGET
    qualify.main()


if __name__ == '__main__': main()
