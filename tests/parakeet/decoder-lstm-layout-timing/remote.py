"""Reuse the original timing supervisor, accounting and bounds."""
from pathlib import Path
import types

TOOLS = Path(__file__).resolve().parent
source = TOOLS/'remote_base.py'
if not source.exists(): source = TOOLS.parent/'prepared-recurrence-timing-amd/remote.py'
text = source.read_text()
before = "for role in ['selected','candidate']:"
assert text.count(before) == 1
text = text.replace(before, "for role in spec['identities']:")
worker = types.ModuleType('layout_timing_supervisor'); worker.__file__ = str(source)
exec(compile(text, str(source), 'exec'), worker.__dict__)
worker.BASE = TOOLS.parent; worker.PROJECT = worker.BASE/'source/Timing.csproj'
idle, live, main = worker.idle, worker.live, worker.main
if __name__ == '__main__': raise SystemExit(main())
