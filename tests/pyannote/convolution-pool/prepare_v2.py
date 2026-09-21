"""Preserve the ordinal failure and rerun unchanged gates on a source-layout successor."""
import common

failure = common.BASE / 'failure-closed.json'
assert common.pin(failure)['sha256'] == '24aa34aa6fb3217c2cb1fc0555061ffb63778a14551651400e7a7e77ff25e8e7'
closed = common.read(failure)
assert not closed['passed'] and not closed['inference_started']
common.verify(closed['files'])
for identity in closed['identities']:
    common.terminal(identity)
common.BASE = common.ROOT / 'artifacts/pyannote-convolution-pool-v2-20260921'
common.monitor.BASE = common.BASE
source = (common.TOOLS / 'prepare.py').read_text(encoding='utf8')
assert source.count('from modify_source import modify') == 1
source = source.replace('from modify_source import modify', 'from modify_source_v2 import modify')
old = 'files = {rel(receipt): pin(receipt), rel(MONITOR): pin(MONITOR)}'
assert source.count(old) == 1
source = source.replace(old, 'files = {rel(receipt): pin(receipt), rel(MONITOR): pin(MONITOR), rel(FAILURE): pin(FAILURE)}')
namespace = dict(__name__='source_layout_successor', __file__=str(common.TOOLS / 'prepare.py'), FAILURE=failure)
exec(compile(source, namespace['__file__'], 'exec'), namespace)
namespace['main']()
