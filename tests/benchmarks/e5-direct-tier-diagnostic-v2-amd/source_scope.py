"""Four unchanged hooks and a fixed diagnostic extension; recover every original byte."""
from pathlib import Path

HOOKS = [
    ('var setup = Stopwatch.StartNew();', 'ClockProbe.Initialize(output, key, mode);\nvar setup = Stopwatch.StartNew();'),
    ('    long start = Stopwatch.GetTimestamp();', '    ClockProbe.Begin(index);\n    long start = Stopwatch.GetTimestamp();'),
    ('    long end = Stopwatch.GetTimestamp();', '    long end = Stopwatch.GetTimestamp();\n    ClockProbe.End(index, start, end);'),
]
SAVE = 'ClockProbe.Save(output);\n'
BOUND = ('mode == "verify" ? 3 : 780', 'mode == "verify" ? 3 : 6000')


def instrument(source):
    result = source
    for old, new in HOOKS:
        assert result.count(old) == 1 and new not in result
        result = result.replace(old, new)
    assert result.count(BOUND[0]) == 1
    result = result.replace(*BOUND)
    result += SAVE
    verify(source, result)
    return result


def verify(source, actual):
    assert actual.endswith(SAVE)
    result = actual[:-len(SAVE)]
    for old, new in reversed(HOOKS):
        assert result.count(new) == 1
        result = result.replace(new, old)
    assert result.count(BOUND[1]) == 1
    result = result.replace(BOUND[1], BOUND[0])
    assert result == source, 'Original consumer changed outside four hooks and the diagnostic extent'
    return dict(passed=True, original_recovered_exactly=True, observation_hooks=4, original_calls=780, diagnostic_calls=6000)
