"""Run original clock, complete-public-result and graph refusal tests unchanged."""
import importlib.util
import unittest
from prepare import PARENT
from consumer_scope import verify_scope


def load_tests(loader, tests, pattern):
    verify_scope()
    for name in ['test_admission', 'test_semantics', 'test_graph_prerequisite']:
        spec = importlib.util.spec_from_file_location('inherited_'+name, PARENT/(name+'.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        tests.addTests(loader.loadTestsFromModule(module))
    return tests


if __name__ == '__main__': unittest.main()
