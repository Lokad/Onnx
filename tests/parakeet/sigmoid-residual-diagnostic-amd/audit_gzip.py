"""Read the retained exporter's gzip format without rerunning its capture."""
import gzip
from run import BASE, TOOLS, pin, read, write

original = TOOLS/'audit.py'
before = "(folder/'events/events.jsonl').read_text().splitlines()"
after = "gzip.decompress((folder/'events/events.jsonl.gz').read_bytes()).decode('utf8').splitlines()"
source = original.read_text(encoding='utf8')
assert source.count(before) == 1
failure = read(BASE/'audit-failed.json')
assert failure['original_auditor'] == pin(original)
assert read(BASE/'collected/state.json')['code'] == 0
assert not (BASE/'closed.json').exists()
write(BASE/'audit-reader-repair.json', dict(original=pin(original), adapter=pin(__file__),
    failure=pin(BASE/'audit-failed.json'), replacement=[before, after],
    capture_repeated=False, checks_relaxed=False,
    reason='The reused qualified exporter emits events.jsonl.gz; the initial reader expected plain JSONL.'))
namespace = dict(__name__='__main__', __file__=str(original), gzip=gzip)
exec(compile(source.replace(before, after), str(original)+'[gzip-reader]', 'exec'), namespace)
