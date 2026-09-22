"""Retain reviewable qualification observations without making a speed claim."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def main():
    target = TOOLS/'observations-20260922.json'; assert not target.exists()
    specs = [
        ('focused', 'pyannote-blocked-spatial-composition-review-v3-20260922', '23816c9ce3b718bca1a89c81e249c382158f05ef434f4dd6d909bcfb060a1d8a'),
        ('layers', 'pyannote-blocked-spatial-layer-graphs-20260922', '3edddec577c52d31d1ec97b952e3ddbd894b26dbe902835cccde958bf59209b4'),
        ('normal', 'pyannote-blocked-spatial-composition-v3-20260922', 'e7a9a30d88ef2a425c0c51b007e2b7d89428d54e50ef7dc4e446e191f95719e3'),
        ('package', 'pyannote-blocked-spatial-package-20260922', 'fc4f4811032ff38ea837b8d53ea68325a63cd9959b7e5d61b9fbb89ba7fefc66')]
    closures = {}; observations = {}
    for name, folder, sha in specs:
        base = ROOT/'artifacts'/folder; closed = base/'closed.json'
        assert pin(closed)['sha256'] == sha
        proof = read(closed); assert proof['passed']
        for relative, wanted in proof['files'].items(): assert pin(base/relative) == wanted, relative
        closures[name] = dict(path=closed.relative_to(ROOT).as_posix(), **pin(closed))
        value = read(base/'analysis.json')
        if name == 'focused':
            review = value['source_review']; core = review['core']
            observations[name] = dict(core=value['core'], data=value['data'], suites=value['suites'], resources=value['resources'],
                source_review=dict(unchanged_core=core['unchanged'], changed_core=core['changed'], added_core=core['added'],
                    generated_method_renames=core['renames'], generated_symbol_mappings=core['symbols'],
                    unchanged_data=review['data_unchanged'], component_methods_equal=review['component_methods_equal'],
                    public_surface_equal=review['public_surface_equal']))
        else: observations[name] = value
        if name == 'layers': observations['layer_results'] = read(base/'output/256.json')
    failures = []
    for folder in ['pyannote-blocked-spatial-composition-20260922', 'pyannote-blocked-spatial-composition-review-v2-20260922',
            'pyannote-blocked-spatial-regressions-20260922', 'pyannote-blocked-spatial-regressions-v2-20260922',
            'pyannote-blocked-spatial-composition-v2-20260922']:
        path = ROOT/'artifacts'/folder/'failure-closed.json'; proof = read(path)
        assert not proof['passed'] and proof['retained_failure']
        failures.append(dict(path=path.relative_to(ROOT).as_posix(), **pin(path)))
    output = dict(passed=True, scope='Local normal-source, focused, captured-layer, full-suite and package qualification',
        selected_for_production=False, full_models_qualified=False, performance_measured=False,
        closures=closures, observations=observations, retained_failures=failures)
    target.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(output=pin(target), closures=closures, retained_failures=failures)))


if __name__ == '__main__': main()
