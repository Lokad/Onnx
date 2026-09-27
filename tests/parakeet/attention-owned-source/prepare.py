"""One owned-preparation policy extension over the qualified Parakeet release."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-attention-owned-source-20260928'
QUALIFIED = ROOT/'artifacts/parakeet-pointwise-tail-root-amd-20260927'
DIAGNOSIS = ROOT/'artifacts/parakeet-attention-cost-amd-20260927'
TARGET = 'src/Lokad.Onnx/GraphOwnedPacking.cs'
TEST = 'tests/Lokad.Onnx.Backend.Tests/OwnedAttentionPreparationTests.cs'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def write(path,value):
    with path.open('x',encoding='utf8') as stream: json.dump(value,stream,indent=2);stream.write('\n')


def change(raw):
    before = raw.decode().replace('\r\n','\n')
    pairs = [
        ('Explicit opt-in for the supported Parakeet feed-forward matrix shapes and node names.',
         'Explicit opt-in for the supported Parakeet feed-forward and square attention matrix shapes and node names.'),
        ('var uses = new Dictionary<string, (int Count, bool MatMulB)>(StringComparer.Ordinal);',
         'var uses = new Dictionary<string, (int Count, bool FeedForward, bool Attention)>(StringComparer.Ordinal);'),
        ('''                    bool eligible = node.Op == OpType.MatMul && Node.IsStandardDomain(node.Domain) && i == 1 && inputs.Length == 2
                        && node.Name?.Contains("/feed_forward", StringComparison.Ordinal) == true;
                    uses.TryGetValue(name, out var prior);
                    uses[name] = (prior.Count + 1, eligible && prior.Count == 0);''',
         '''                    bool eligible = node.Op == OpType.MatMul && Node.IsStandardDomain(node.Domain) && i == 1 && inputs.Length == 2;
                    bool attention = node.Name is string consumer &&
                        (consumer.EndsWith("/self_attn/linear_q/MatMul", StringComparison.Ordinal)
                        || consumer.EndsWith("/self_attn/linear_k/MatMul", StringComparison.Ordinal)
                        || consumer.EndsWith("/self_attn/linear_v/MatMul", StringComparison.Ordinal)
                        || consumer.EndsWith("/self_attn/linear_out/MatMul", StringComparison.Ordinal)
                        || consumer.EndsWith("/self_attn/linear_pos/MatMul", StringComparison.Ordinal));
                    uses.TryGetValue(name, out var prior);
                    bool single = eligible && prior.Count == 0;
                    uses[name] = (prior.Count + 1,
                        single && node.Name?.Contains("/feed_forward", StringComparison.Ordinal) == true,
                        single && attention);'''),
        ('if (!uses.TryGetValue(name, out var use) || use.Count != 1 || !use.MatMulB',
         'if (!uses.TryGetValue(name, out var use) || use.Count != 1 || !(use.FeedForward || use.Attention)'),
        ('                if (!((n == 1024 && k == 4096) || (n == 4096 && k == 1024))) continue;',
         '''                if (!((use.FeedForward && ((n == 1024 && k == 4096) || (n == 4096 && k == 1024)))
                    || (use.Attention && n == 1024 && k == 1024))) continue;''')]
    after = before
    for old,new in pairs:
        assert after.count(old)==1,old
        after=after.replace(old,new)
    restored=after
    for old,new in reversed(pairs):
        assert restored.count(new)==1
        restored=restored.replace(new,old)
    assert restored==before
    data=(after.replace('\n','\r\n') if b'\r\n' in raw else after).encode()
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=TARGET,tofile=TARGET))
    return data,patch


def references():
    assert pin(QUALIFIED/'closed.json')['sha256']=='fc11676361a50ef613f783fcb51e4ead488c9c2ee34a3edddcbb561f67037e47'
    closure=read(QUALIFIED/'closed.json')
    assert closure['passed'] and closure['files']['bundle/stage.json']==pin(QUALIFIED/'bundle/stage.json')
    stage=read(QUALIFIED/'bundle/stage.json')
    source={n.removeprefix('source/'):v for n,v in stage['files'].items() if n.startswith('source/')}
    assert len(source)==445
    for name,wanted in source.items():
        assert pin(QUALIFIED/'bundle/source'/name)==pin(ROOT/name)==wanted,name
    assert pin(DIAGNOSIS/'closed.json')['sha256']=='5ff17ed5e3e9596789bf27b27abc01584342b08f81005a4a25643b0df045b5f2'
    diagnosis=read(DIAGNOSIS/'closed.json')
    assert diagnosis['passed'] and diagnosis['usable_for_candidate_selection']
    assert diagnosis['analysis']==pin(DIAGNOSIS/'analysis.json')
    return source


def main():
    assert not BASE.exists()
    original=references()
    values={n:(QUALIFIED/'bundle/source'/n).read_bytes() for n in original}
    values[TARGET],patch=change(values[TARGET])
    assert TEST not in values
    values[TEST]=(TOOLS/'OwnedAttentionPreparationTests.cs.txt').read_bytes()
    BASE.mkdir()
    for name,data in values.items():
        path=BASE/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    (BASE/'candidate.patch').write_text(patch,encoding='utf8')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'PLAN.md').read_bytes())
    source={n:pin(BASE/'source'/n) for n in values}
    assert [n for n in original if source[n]!=original[n]]==[TARGET]
    write(BASE/'prepared.json',dict(passed=True,release_admitted=False,root_product_changed=False,
        baseline=pin(QUALIFIED/'closed.json'),diagnosis=pin(DIAGNOSIS/'closed.json'),
        source=source,source_before=original,changed_product_files=[TARGET],added_product_files=[],
        changed_methods=['PrepareOwnedMatMulWeights'],added_methods=[],added_tests=[TEST],
        source_reversible=True,arithmetic_leaves_unchanged=True,expected_added_attention_weights=92,
        expected_retained_maps=37,maximum_packed_bytes=268435456,
        patch=pin(BASE/'candidate.patch'),plan=pin(BASE/'prospective-plan.md'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),files=len(source),changed=[TARGET],added_tests=[TEST])))


if __name__=='__main__':main()
