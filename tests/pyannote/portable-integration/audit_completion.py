"""Apply the original method/package gates to the closed partial run and continuation."""
from resume import *


def main():
    original = TOOLS / 'audit.py'
    source = original.read_text(encoding='utf8')
    def replace(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    replace('from common import *', 'from resume import *')
    replace("inputs = read(BASE / 'source-inputs.json')\n    verify(inputs['files'])", '''inputs = read(PRIOR / 'source-inputs.json')
    snapshots = read(PRIOR / 'root-input-snapshots.json')
    for name, wanted in inputs['files'].items():
        path = ROOT / (snapshots[name]['path'] if name in snapshots else name)
        assert pin(path) == wanted, name''')
    replace("source, runtime = BASE / 'source', BASE / 'runtime'", "source, runtime = PRIOR / 'source', PRIOR / 'runtime'")
    replace("assert pin(path) == pin(ROOT / path.relative_to(source))",
        "assert pin(path) == pin(ROOT / snapshots[path.relative_to(source).as_posix()]['path'])")
    replace("instructions = read(BASE / 'instructions.json')", "instructions = read(PRIOR / 'instructions.json')")
    replace('suites = [suite(*args) for args in expected]',
        "suites = [read_suite(BASE if args[0] == 'tensors-full' else PRIOR, *args) for args in expected]")
    start, end = source.index('    jobs = ['), source.index("    package = BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg'")
    source = source[:start] + '''    predecessor = read(PRIOR / 'failure-closed.json')
    assert not predecessor['passed'] and predecessor['checker_refusal'] and predecessor['numerical_failures'] == 0
    verify(predecessor['files'])
    earlier = resources(PRIOR, ['cli-restore', 'cli-build', 'backend-restore', 'backend-build', 'tensors-restore', 'tensors-build',
        'bridge-restore', 'bridge-build', 'instructions', 'focused', 'hardware-disabled', 'backend-full'], 1)
    later = resources(BASE, ['tensors-full', 'package', 'consumer-restore', 'consumer-build', 'consumer'], 0)
    identities = earlier['identities'] + later['identities']
    all_resources = earlier['resources'] + later['resources']
''' + source[end:]
    replace("resources=resources, resource_samples=sum(r['samples'] for r in resources), peak_rss=max(r['peak_rss'] for r in resources),",
        "resources=all_resources, resource_samples=sum(r['samples'] for r in all_resources), peak_rss=max(r['peak_rss'] for r in all_resources),")
    namespace = dict(globals(), __name__='independent_completion_auditor', __file__=str(Path(__file__)))
    exec(compile(source, str(original) + ' [closed continuation]', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
