"""Exact, reviewable adaptations of the already qualified operator consumers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
CONTROL_CORE = '1279b4b662241db2404fa4875eae20eaa924f15677a85655f99b8f81cd24b309'
CORE = 'abbf5e9878aeccf929a327ef4c1c3f5c9ff8afb8d9dec7ddb80455636cd1684d'
DATA = 'eb452663a09daa5d287fff1f65c07f2105f5c10ab474f845f210b2b30e3b921f'


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def caller_source():
    source = (ROOT / 'artifacts/pyannote-single-panel-composition-20260922/caller/Program.cs').read_text(encoding='utf8')
    return replace_once(source, 'e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838', CONTROL_CORE)


def probe_source():
    source = (ROOT / 'tests/parakeet/reduction-dispatch/Probe.cs').read_text(encoding='utf8')
    source = replace_once(source, 'args.Length == 4, "root, original runtime, result path, normal|disabled"',
                          'args.Length == 5, "root, original runtime, result path, normal|disabled, expected runtime"')
    source = replace_once(source, 'Environment.Version.ToString() == "10.0.12"',
                          '(args[4] is "10.0.12" or "10.0.8") && Environment.Version.ToString() == args[4]')
    source = replace_once(source, 'Require(!Avx512F.IsSupported, "This bounded local comparison is AVX2; AMD is separate");',
                          'Require(Avx2.IsSupported && Fma.IsSupported, "Normal AVX2/FMA required");')
    source = replace_once(source, '        int tests = 0,', '        var precedence = new List<object>();\n        int tests = 0,')
    source = replace_once(source, '            var rng = new Random(652199);',
                          '            if (Avx512F.IsSupported) CheckPrecedence(old.Assembly, prepared, admit, precedence);\n            var rng = new Random(652199);')
    source = replace_once(source, '            runtime = Environment.Version.ToString(),',
                          '            prepared_precedence = precedence, avx512 = Avx512F.IsSupported,\n            runtime = Environment.Version.ToString(),')
    return replace_once(source, '    static void Main(string[] args)',
                        (TOOLS / 'Precedence.cs.txt').read_text(encoding='utf8') + '\n    static void Main(string[] args)')


def bridge_source():
    source = (ROOT / 'tests/pyannote/combined-avx512/Inventory.cs.txt').read_text(encoding='utf8')
    return replace_once(source, 'new[] { "Lokad.Onnx.dll", "Lokad.Onnx.Data.dll" }', 'new[] { "Caller.dll" }')
