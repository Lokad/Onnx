"""Draft diagnostic-only stage boundaries; never modify product or prepared inputs.

Removing each exact edit restores the complete original file. The emitted sources
are not a qualified runtime: VM compilation, code review and observer controls
remain prerequisites to using their measurements.
"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/parakeet-feed-forward-cost-source-20260925'
STAGES = {
    'ScratchRent': 1000,
    'WeightPacking': 1001,
    'PackedMultiplication': 1002,
    'ScratchReturn': 1003,
    'OddRowRemainder': 1004,
    'OutputPreparation': 1005,
    'PreparedMultiplication': 1006,
}


def pin(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def marker(stage, indent):
    value = 'OpStage.Math' if stage == 'Math' else 'FeedForwardCostStages.' + stage
    return ' ' * indent + f'Profiler.StartOpStage({value});\n'


def transform(name, original):
    """Only add timing boundaries; retain every original statement and flag."""
    content = original
    edits = []

    def replace(before, after):
        nonlocal content
        assert content.count(before) == 1, (name, before)
        assert before != after
        content = content.replace(before, after)
        edits.append(dict(before=before, after=after))

    def surround(line, stage, indent, finish='Math'):
        replace(line, marker(stage, indent) + line + marker(finish, indent))

    if name == 'Zzz.IsolatedShortMatMul.cs':
        surround('        float[] packed = RentScratch<float>(n * k, options);\n', 'ScratchRent', 8)
        surround('                ShortWidePackPanelsB(n, k, y, pp);\n', 'WeightPacking', 16,
                 finish='PackedMultiplication')
        surround('            ArrayPool<float>.Shared.Return(packed);\n', 'ScratchReturn', 12)
        tail = ('        if (rows != m)\n'
                '            ShortWideMultiplyRemainder(1, n, k, x + rows * n, y, output + rows * k);\n')
        surround(tail, 'OddRowRemainder', 8)
    elif name == 'TensorOps.MatMul.cs':
        surround('        if (clearDestination) target.Buffer.Span.Clear();\n', 'OutputPreparation', 8)
        surround('        var destination = new DenseTensor<float>(new Memory<float>(pool.RentCleared<float>((int)length)), dims);\n',
                 'OutputPreparation', 8)
        surround('            var z = DenseTensor<float>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());\n',
                 'OutputPreparation', 12)
        # Observe the call boundary rather than changing this aggressively inlined
        # dispatcher or any arithmetic leaf. Both original batch branches remain.
        surround('                        RunPreparedPackedRows(m, n, k, (float*)xp0 + xOff[bi], pp, (float*)zp0 + zOff[bi]);\n',
                 'PreparedMultiplication', 24)
        surround('                    RunPreparedPackedRows(m, n, k, xp + ox, pp, zp + oz);\n',
                 'PreparedMultiplication', 20)
    elif name == 'Zzz.WideProjectionEntry.cs':
        surround('        if (clearDestination) destination.Buffer.Span.Clear();\n', 'OutputPreparation', 8)
        surround('                RunPreparedPackedRows(m, n, k, (float*)xh.Pointer, (float*)ph.Pointer, (float*)oh.Pointer);\n',
                 'PreparedMultiplication', 16)
    elif name == 'Profiler.cs':
        # Existing stage timing omits transition overhead. Also record enclosing
        # node intervals so the analysis must expose, rather than subtract, that
        # overhead plus any uncovered work. The original wall-only path is intact.
        before = ('                AddTimeLocked();\n'
                  '                Profile.Push(new NodeProfile()')
        after = ('                Wall.Add(new WallNode() { NodeId = id, Op = op, StartTicks = System.Diagnostics.Stopwatch.GetTimestamp() });\n'
                 '                wallOpen = Wall.Count - 1;\n' + before)
        replace(before, after)
        before = ('                AddTimeLocked();\n'
                  '            }\n'
                  '        }\n\n'
                  '        [MethodImpl(MethodImplOptions.AggressiveInlining)]\n'
                  '        public void StartOpStage')
        after = ('                AddTimeLocked();\n'
                 '                if (wallOpen >= 0)\n'
                 '                {\n'
                 '                    var completed = Wall[wallOpen];\n'
                 '                    completed.EndTicks = System.Diagnostics.Stopwatch.GetTimestamp();\n'
                 '                    Wall[wallOpen] = completed;\n'
                 '                    wallOpen = -1;\n'
                 '                }\n'
                 '            }\n'
                 '        }\n\n'
                 '        [MethodImpl(MethodImplOptions.AggressiveInlining)]\n'
                 '        public void StartOpStage')
        replace(before, after)
    else:
        raise ValueError(name)

    restored = content
    for edit in reversed(edits):
        assert restored.count(edit['after']) == 1
        restored = restored.replace(edit['after'], edit['before'])
    assert restored == original, name
    return content, edits


def main():
    assert not OUT.exists(), 'Preserve existing review; do not replace it'
    prior_path = ROOT / 'tests/parakeet/slice-dense-conversion-results/feed-forward-review-20260925.json'
    prior = json.loads(prior_path.read_text(encoding='utf8'))
    assert prior['passed'] and prior['packing_time_not_yet_separated']
    sources = ['Zzz.IsolatedShortMatMul.cs', 'TensorOps.MatMul.cs',
               'Zzz.WideProjectionEntry.cs', 'Profiler.cs']
    prepared = []
    for name in sources:
        path = ROOT / 'src/Lokad.Onnx' / name
        if name in prior['source_dispatch_unchanged']:
            assert pin(path) == prior['source_dispatch_unchanged'][name], name
        original = path.read_text(encoding='utf8')
        changed, edits = transform(name, original)
        prepared.append((name, path, original, changed, edits))
    assert sum(len(v[4]) for v in prepared) == 13
    # All work above is read-only. Create the bounded review after every anchor
    # and exact inverse has passed. No source tree or deployment is modified.
    OUT.mkdir()
    patches = []
    inventory = {}
    for name, path, original, changed, edits in prepared:
        output = OUT / name
        with output.open('x', encoding='utf8', newline='\n') as stream:
            stream.write(changed)
        inventory[name] = dict(original=pin(path), diagnostic=pin(output), edits=edits)
        patches.extend(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                          fromfile='original/' + name, tofile='diagnostic/' + name))
    stage_source = ('namespace Lokad.Onnx;\n\n// Isolated diagnostic vocabulary; never part of the package.\n'
                    'internal static class FeedForwardCostStages\n{\n' +
                    ''.join(f'    internal const OpStage {name} = (OpStage){number};\n' for name, number in STAGES.items()) +
                    '}\n')
    with (OUT / 'FeedForwardCostStages.cs').open('x', encoding='utf8', newline='\n') as stream:
        stream.write(stage_source)
    with (OUT / 'diagnostic.patch').open('x', encoding='utf8', newline='\n') as stream:
        stream.write(''.join(patches))
    review = dict(passed=True, scope='source-transformation-only', compiled=False, executed=False,
                  product_changed=False, no_candidate_selected=True,
                  prior_feed_forward_review=pin(prior_path), files=inventory,
                  stage_vocabulary=STAGES, added_source=pin(OUT / 'FeedForwardCostStages.cs'),
                  patch=pin(OUT / 'diagnostic.patch'), generator=pin(Path(__file__)),
                  controls_required=['original Core, phase-only graph clock',
                                     'original Core, existing stage profiler',
                                     'diagnostic Core, stage profiler and node intervals'],
                  limitations=['Stage transitions omit their own clock/locking overhead; expose the residual.',
                               'Profiler lock serializes concurrent workers; use only the qualified single-thread workload.',
                               'Prepared and dynamic multiplication stages include dispatch and all selected arithmetic.',
                               'Packing and multiplication stage values include profiling and process effects.',
                               'Graph scalar multiplication must be joined with the corresponding ORT fused projection.',
                               'No inference result or performance conclusion exists for this diagnostic yet.'])
    with (OUT / 'review.json').open('x', encoding='utf8') as stream:
        json.dump(review, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, files=len(inventory), edits=13,
                         review=pin(OUT / 'review.json'), compiled=False, executed=False)))


if __name__ == '__main__':
    main()
