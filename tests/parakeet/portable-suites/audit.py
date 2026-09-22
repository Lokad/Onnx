"""Close normal source/test/package evidence after its controller exits."""
from run import BASE, read, verify, suite
from common import resources, close


def main():
    proof = read(BASE / 'verified.json'); assert proof['passed']; verify(proof['files'])
    assert proof['suites'] == [suite('backend', 3290, 93), suite('tensors', 343, 0)]
    jobs = {name + '-' + stage: (8, 8, 900, False) for name in ['cli', 'backend', 'tensors'] for stage in ['restore', 'build']}
    jobs.update({name: (minimum, 8, 900, False) for name, minimum in [('instructions', 8), ('backend-tests', 10), ('tensors-tests', 10),
        ('package', 8), ('consumer-restore', 8), ('consumer-build', 8), ('consumer', 10)]})
    observed = resources(BASE, 'processes.json', jobs)
    analysis = {k: v for k, v in proof.items() if k != 'files'}
    analysis.update(observed)
    close(BASE, analysis, proof['files'], observed['identities'])
    print(proof['suites'], flush=True)


if __name__ == '__main__': main()
