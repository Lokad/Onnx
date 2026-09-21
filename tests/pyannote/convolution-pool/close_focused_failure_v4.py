"""Preserve the fixture's unsupported post-failure cache-hit expectation."""
from pathlib import Path
path = Path(__file__).with_name('close_focused_failure.py')
source = path.read_text(encoding='utf8')
source = source.replace('pyannote-convolution-pool-v3-20260921', 'pyannote-convolution-pool-v4-20260921')
source = source.replace('Synthetic graph omits IsFused marker and descriptors needed to keep output declarations stable across Reset; all direct destination tests and method-isolation checks pass.',
    'Fixture requires a cache hit immediately after injected failure consumes the only cached array and returns none. Separate recorded diagnostics establish reuse 32/0/32 bytes before failure, first recovery and next recovery. Product behavior is correct; preserve all numerical/ownership assertions and require restored reuse after successful recovery.')
exec(compile(source, str(path), 'exec'), dict(__name__='__main__', __file__=str(path)))
