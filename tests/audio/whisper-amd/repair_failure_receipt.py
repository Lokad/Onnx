"""Correct only the final-log hash in the failed campaign's closure inventory."""
import datetime
import json
from common import *


def main():
    old_path = BASE/'failure-closed.json'; target = BASE/'failure-closed-v2.json'
    assert not target.exists()
    old = read(old_path); assert old['closure_passed'] is True and old['campaign_passed'] is False and old['reason'] == 'disk guard'
    changed = {name: dict(expected=wanted, actual=pin(ROOT/name)) for name, wanted in old['files'].items() if pin(ROOT/name) != wanted}
    key = (BASE/'failure-close.log').relative_to(ROOT).as_posix()
    assert set(changed) == {key} and changed[key]['expected'] == dict(bytes=0, sha256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
    log = json.loads((ROOT/key).read_text()); assert log['closure'] == pin(old_path) and log['closure_passed'] is True and log['campaign_passed'] is False
    assert ssh(PRELUDE+'terminal(%r)\nprint("terminal")\n' % old['births']).strip() == 'terminal'
    repaired = dict(old); repaired['files'] = dict(old['files'])
    repaired['files'][key] = changed[key]['actual']
    repaired['files'][old_path.relative_to(ROOT).as_posix()] = pin(old_path)
    repaired['files'][Path(__file__).resolve().relative_to(ROOT).as_posix()] = pin(Path(__file__))
    repaired['repair'] = dict(reason='final stdout log appended after initial inventory', changed=changed,
                              original=pin(old_path), utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(target, repaired)
    for name, wanted in read(target)['files'].items():
        assert pin(ROOT/name) == wanted, name
    print(json.dumps(dict(closure_passed=True, campaign_passed=False, pins=len(repaired['files']), closure=pin(target))))


if __name__ == '__main__':
    main()
