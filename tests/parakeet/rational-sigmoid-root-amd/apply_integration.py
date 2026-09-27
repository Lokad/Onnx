"""Apply the fixed arithmetic product and portable tests only after fresh admission."""
import shutil
from prepare import gates, evidence_spec
from source_scope import ROOT, SOURCE, FIXTURE, APPLIED, TARGET, HELPER, TEST, CHANGED, root_files, verify_root
from protocol import pin, save


def main():
    assert not APPLIED.exists()
    source = gates(); verify_root(source['before'])
    assert not (ROOT/HELPER).exists() and not (ROOT/TEST).exists()
    expected = root_files(source); spec = evidence_spec()
    APPLIED.mkdir(); (APPLIED/'before').mkdir()
    shutil.copy2(ROOT/TARGET, APPLIED/'before/CPUExecutionProvider.Elementwise.cs')
    for name, path in [(TARGET,SOURCE/'source'/TARGET), (HELPER,SOURCE/'source'/HELPER),
                       (TEST,FIXTURE/'SigmoidVectorTests.cs')]:
        assert pin(path) == expected[name]; shutil.copy2(path, ROOT/name)
    verify_root(expected)
    save(APPLIED/'applied.json', dict(passed=True, root_build_pending=True, prepared=spec['source_prepared'],
        fixture=pin(FIXTURE/'prepared.json'), source_files=expected, changed=CHANGED,
        before={TARGET: source['before'][TARGET]},
        prerequisites={k:v['closed'] for k,v in spec['prerequisites'].items()},
        graph_qualification=spec['graph_qualification']['closed']))
    print(dict(passed=True, source_files=len(expected), changed=3, receipt=pin(APPLIED/'applied.json')))


if __name__ == '__main__': main()
