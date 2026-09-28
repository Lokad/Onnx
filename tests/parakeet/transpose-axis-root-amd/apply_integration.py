"""Apply the measured transpose dispatch and already qualified portable tests only after fresh release admission."""
import shutil
from prepare import gates, evidence_spec
from source_scope import ROOT, SOURCE, FIXTURE, APPLIED, TARGETS, TEST, CHANGED, root_files, verify_root
from protocol import pin, save


def main():
    assert not APPLIED.exists()
    source = gates(); verify_root(source['before'])
    assert not (ROOT/TEST).exists()
    expected = root_files(source); spec = evidence_spec()
    APPLIED.mkdir(); (APPLIED/'before').mkdir()
    for name in TARGETS: shutil.copy2(ROOT/name, APPLIED/'before'/name.rsplit('/',1)[1])
    inputs = [(name,SOURCE/'source'/name) for name in TARGETS]
    inputs.append((TEST,FIXTURE))
    for name, path in inputs:
        assert pin(path) == expected[name]; shutil.copy2(path, ROOT/name)
    verify_root(expected)
    save(APPLIED/'applied.json', dict(passed=True, root_build_pending=True, prepared=spec['source_prepared'],
        fixture=pin(FIXTURE), source_files=expected, changed=CHANGED,
        before={name:source['before'][name] for name in TARGETS},
        prerequisites={k:v['closed'] for k,v in spec['prerequisites'].items()},
        graph_qualification=spec['graph_qualification']['closed']))
    print(dict(passed=True, source_files=len(expected), changed=len(CHANGED), receipt=pin(APPLIED/'applied.json')))


if __name__ == '__main__': main()
