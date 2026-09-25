"""Four explicit observation hooks; reversing them must recover every original byte."""
from pathlib import Path

HOOKS = [
    ('var setup = Stopwatch.StartNew();', 'ClockProbe.Initialize(output, key, mode);\nvar setup = Stopwatch.StartNew();'),
    ('    long start = Stopwatch.GetTimestamp();', '    ClockProbe.Begin(index);\n    long start = Stopwatch.GetTimestamp();'),
    ('    long end = Stopwatch.GetTimestamp();', '    long end = Stopwatch.GetTimestamp();\n    ClockProbe.End(index, start, end);'),
]
SAVE = 'ClockProbe.Save(output);\n'


def instrument(source):
    result = source
    for old, new in HOOKS:
        assert result.count(old) == 1 and new not in result
        result = result.replace(old, new)
    result += SAVE
    verify(source, result)
    return result


def verify(source, actual):
    assert actual.endswith(SAVE)
    result = actual[:-len(SAVE)]
    for old, new in reversed(HOOKS):
        assert result.count(new) == 1
        result = result.replace(new, old)
    assert result == source, 'Original consumer changed outside the four hooks'
    return dict(passed=True, original_recovered_exactly=True, observation_hooks=4)
