"""Apply only the test fixture repair and retain the original failed integration."""
import shutil
from prepare import gates, evidence_spec
from source_scope import (ROOT, FIXTURE, APPLIED, PREVIOUS_APPLIED, FAILED,
                          TEST, CHANGED, root_files, verify_root)
from protocol import pin, read, save


def main():
    assert not APPLIED.exists()
    source = gates(); verify_root(source['source'])
    previous = read(PREVIOUS_APPLIED/'applied.json')
    assert previous['passed'] and previous['source_files'] == source['source']
    assert previous['changed'] == CHANGED
    spec = evidence_spec()
    assert previous['prerequisites'] == {k: v['closed'] for k, v in spec['prerequisites'].items()}
    assert previous['graph_qualification'] == spec['graph_qualification']['closed']
    assert (ROOT/TEST).stat().st_nlink == 1
    expected = root_files(source)
    APPLIED.mkdir(); (APPLIED/'before').mkdir()
    shutil.copy2(ROOT/TEST, APPLIED/'before/OwnedAttentionPreparationTests.cs')
    shutil.copy2(FIXTURE, ROOT/TEST)
    verify_root(expected)
    result = dict(previous, source_files=expected, fixture=pin(FIXTURE),
                  previous_integration=pin(PREVIOUS_APPLIED/'applied.json'),
                  failed_run=pin(FAILED/'failed.json'), fixture_repair_only=True,
                  recovery_changed=[TEST])
    save(APPLIED/'applied.json', result)
    print(dict(passed=True, source_files=446, recovery_changed=[TEST], receipt=pin(APPLIED/'applied.json')))


if __name__ == '__main__': main()
