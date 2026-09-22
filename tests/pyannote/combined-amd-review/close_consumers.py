"""Close exact consumer-literal corrections after every build process is terminal."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/pyannote-combined-consumers-20260922'
spec = importlib.util.spec_from_file_location('resource_audit', ROOT / 'tests/pyannote/convolution-portable-qualification/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def main():
    prepared = common.read(BASE / 'prepared.json')
    assert prepared['passed'] and prepared['product_binaries_unchanged'] and prepared['inference_assertions_unchanged']
    common.verify(prepared['files'])
    jobs = {role + '-' + stage: (8, 900, False) for role in ['portable', 'rows', 'bridge'] for stage in ['restore', 'build']}
    jobs.update({role + '-instructions': (8, 900, False) for role in ['portable', 'rows']})
    resources = common.resources(BASE, 'processes.json', jobs)
    for role, report in prepared['reports'].items():
        inventory = common.read(BASE / (role + '-instructions.json'))['observations'][0]
        assert inventory['methods'] == 96 and not inventory['added'] and not inventory['removed']
        assert len(inventory['differences']) == 1
        method = inventory['differences'][0]
        before, after = inventory['normalized_methods'][method], inventory['candidate_methods'][method]
        assert before.count(report['before']) == 1 and before.replace(report['before'], report['after']) == after
        assert inventory['public_surface_equal'] and report['passed']
    analysis = dict(passed=True, reports=prepared['reports'], product_binaries_unchanged=True, **resources)
    common.close(BASE, analysis, prepared['files'], resources['identities'])


if __name__ == '__main__':
    main()
