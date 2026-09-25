"""Retain existing profiler clocks and repeat the exact counter loop four times."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'owned-packed-weight-counters-v2/Program.cs.txt'
EDITS = [
    ('new JsonSerializerOptions { WriteIndented = true }',
     'new JsonSerializerOptions { WriteIndented = false }'),
    ('        Require(!Directory.Exists(output), "Fresh output");',
     '        Require(mode == "512", "Normal-mode diagnostic matches application timing");\n'
     '        Require(!Directory.Exists(output), "Fresh output");'),
    ('        for (int index = 0; index < cases.Length; index++)',
     '        for (int pass = 0; pass < 4; pass++)\n'
     '        for (int index = 0; index < cases.Length; index++)'),
    ('                using (Profiler.BeginExecution(true)) encoded = Execute(enc, new() { ["audio_signal"] = features["features"], ["length"] = features["features_lens"] });',
     '                long encoderStarted = Stopwatch.GetTimestamp();\n'
     '                using (Profiler.BeginExecution(true)) encoded = Execute(enc, new() { ["audio_signal"] = features["features"], ["length"] = features["features_lens"] });\n'
     '                long encoderEnded = Stopwatch.GetTimestamp();'),
    ('                var byId = profile.ToDictionary(p => p.NodeId); var calls = new List<object>();',
     '                Require(profile.All(p => p.OpsProfile.All(s => s.Time.Ticks >= 0)), "Nonnegative stage clocks");\n'
     '                var byId = profile.ToDictionary(p => p.NodeId); var calls = new List<object>();'),
    ('                    calls.Add(new { node = node.Name, weight = node.Inputs[1], a, b, owned, copy_y_stages = copies });',
     '                    calls.Add(new { node = node.Name, node_id = node.ID, weight = node.Inputs[1], a, b, owned, copy_y_stages = copies,\n'
     '                        copy_y_ticks = p.OpsProfile.Where(s => s.Stage == OpStage.CopyY).Sum(s => s.Time.Ticks),\n'
     '                        math_ticks = p.OpsProfile.Where(s => s.Stage == OpStage.Math).Sum(s => s.Time.Ticks),\n'
     '                        node_ticks = p.OpsProfile.Sum(s => s.Time.Ticks) });'),
    ('                var record = new { name, frames, remainder, input_hash = inputHash, frontend = featureHashes, encoder = hashes,',
     '                var record = new { name, frames, remainder, pass, phase = pass == 0 ? "warmup" : "measured", request_index = pass * cases.Length + index,\n'
     '                    encoder_start = encoderStarted, encoder_end = encoderEnded, encoder_frequency = Stopwatch.Frequency, profile_frequency = TimeSpan.TicksPerSecond,\n'
     '                    node_times = profile.Select(p => new { node_id = p.NodeId, ticks = p.OpsProfile.Sum(s => s.Time.Ticks) }).ToArray(),\n'
     '                    input_hash = inputHash, frontend = featureHashes, encoder = hashes,'),
    ('                Write(output, $"{index:D3}.json", record); records.Add(record);',
     '                Write(output, $"{pass * cases.Length + index:D3}.json", record); records.Add(record);'),
    ('        Require(held.All(p => Bits(p.Tensor) == p.Hash), "Held outputs independent");',
     '        Require(records.Count == 80, "One warmup and three measured corpus passes");\n'
     '        Require(held.All(p => Bits(p.Tensor) == p.Hash), "Held outputs independent");'),
    ('            product_rebuilt = false, application_scored = false, forced_gc = false,',
     '            product_rebuilt = false, application_scored = false, forced_gc = false, diagnostic_only = true, passes = 4, warmup_passes = 1,\n'
     '            node_manifest = encoder.Nodes.Select(n => new { id = n.ID, name = n.Name, op = n.Op.ToString(), inputs = n.Inputs, outputs = n.Outputs }).ToArray(),'),
]


def make(source):
    actual = source
    for before, after in EDITS:
        assert actual.count(before) == 1 and after not in actual
        actual = actual.replace(before, after)
    verify(source, actual)
    return actual


def verify(source, actual):
    for before, after in reversed(EDITS):
        assert actual.count(after) == 1
        actual = actual.replace(after, before)
    assert actual == source, 'Counter consumer changed beyond loop extent, observations and JSON whitespace'
    return dict(passed=True, original_recovered_exactly=True, edits=len(EDITS),
                products_changed=False, measurement_passes=3, warmup_passes=1)


if __name__ == '__main__':
    source = ORIGINAL.read_text(encoding='utf8')
    output = HERE / 'Program.cs.txt'
    if sys.argv[1:] == ['--write']:
        assert not output.exists()
        with output.open('x', encoding='utf8', newline='\n') as stream:
            stream.write(make(source))
    else:
        assert not sys.argv[1:]
    result = verify(source, output.read_text(encoding='utf8'))
    print(json.dumps(dict(**result, source_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
