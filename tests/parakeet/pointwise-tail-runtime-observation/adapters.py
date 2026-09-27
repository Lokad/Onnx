"""Small explicit adaptations of the already used runtime observer."""
from pathlib import Path

PRIOR = Path(__file__).resolve().parent.parent/'decoder-lstm-runtime-observation'


def adapted(name):
    text = (PRIOR/name).read_text()
    changes = {
        'remote.py': [
            ('ParakeetRecurrenceTiming', 'PointwiseTailRuntime', 3),
            ("BASE, 'selectedfallback', '512', name, BASE/name/'output'",
             "BASE/'spec.json', BASE/'built.json', BASE/name/'output', 'candidate'", 1),
            ("dict(passed=True, files=files, consumer=pin(BASE/'runtime/PointwiseTailRuntime.dll'))",
             "dict(passed=True, files=files, products=spec['products'], consumer=pin(BASE/'runtime/PointwiseTailRuntime.dll'))", 1)],
        'events.py': [('len(markers) == 7600 and len(value[\'diagnostics\']) == 3800',
                       'len(markers) == 800 and len(value[\'diagnostics\']) == 400', 1)],
        'audit.py': [('events=len(events), calls=3800', 'events=len(events), calls=400', 1)],
    }
    for before, after, count in changes[name]:
        assert text.count(before) == count, (name, before, text.count(before))
        text = text.replace(before, after)
    return text
