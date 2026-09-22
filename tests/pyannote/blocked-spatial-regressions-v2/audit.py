"""Reconcile project-based test execution with the unchanged product evidence."""
import importlib.util
from run import BASE, FAILED, PREVIOUS, original, pin, read, verify


def main():
    value = read(BASE/'successor-inputs.json'); verify(value['files'])
    state = read(BASE/'controller.json')
    for row in state['runs']:
        assert row['command'][2].endswith('.csproj')
        assert all(arg in row['command'] for arg in ['--no-build', '--no-restore', '--tl:off'])
    # The previous auditor resolves its imports to the already rebound original.
    import sys
    current = sys.modules['run']; sys.modules['run'] = original
    try:
        spec = importlib.util.spec_from_file_location('original_regression_auditor', PREVIOUS/'audit.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.main()
    finally: sys.modules['run'] = current


if __name__ == '__main__': main()
