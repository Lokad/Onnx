"""Reuse the full timing audit; keep failed execution evidence independently."""
from pathlib import Path
import types
from prepare import BASE, PARENT
from protocol import pin, read, save


def main():
    assert not (BASE/'closed.json').exists() and not (BASE/'failed.json').exists()
    receipt = read(BASE/'collected/collection.json')
    assert receipt['terminal'] and receipt['input_error'] is None
    for name, wanted in receipt['files'].items(): assert pin(BASE/'collected'/name) == wanted, name
    if receipt['code'] != 0:
        save(BASE/'failed.json', dict(passed=False, terminal=True, admitted=False,
            state=read(BASE/'collected/identity.json'),
            files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}))
        raise AssertionError('Retain terminal failure and diagnose; do not replay.')
    source = PARENT/'audit.py'; text = source.read_text()
    changes = {
        'complete_call_clocks=30400,warmup=15200,measured=15200,exact_output_arrays=91200':
            'complete_call_clocks=60800,warmup=30400,measured=30400,exact_output_arrays=182400',
        "source_prepared=pin(collected/'evidence/source-prepared.json')":
            "contract_diagnosis=pin(collected/'evidence/contracts-diagnosis.json')",
    }
    for before, after in changes.items():
        assert text.count(before) == 1, before; text = text.replace(before, after)
    audit = types.ModuleType('layout_timing_audit'); audit.__file__ = str(source)
    exec(compile(text, str(source), 'exec'), audit.__dict__)
    audit.main()


if __name__ == '__main__': main()
