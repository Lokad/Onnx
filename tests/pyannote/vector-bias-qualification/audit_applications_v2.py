"""Use unchanged public/native auditing for the corrected test-harness copy."""
import applications_v2
from common import *


def main():
    namespace = applications_v2.load()
    namespace['__name__'] = 'corrected_application_auditor'
    path = TOOLS / 'audit_applications.py'
    source = path.read_text(encoding='utf8')
    assert source.count('from applications import *') == 1
    source = source.replace('from applications import *', 'from common import *')
    exec(compile(source, str(path), 'exec'), namespace)
    prepared = read(namespace['BASE'] / 'applications-prepared.json')
    assert prepared['files'][rel(namespace['AUDITOR'])] == pin(namespace['AUDITOR'])
    correction = read(namespace['BASE'] / 'cli-test-correction.json')
    assert correction['corrected'] == pin(namespace['BASE'] / 'suites-source' / correction['path'])
    namespace['load']()['main']()


if __name__ == '__main__':
    main()
