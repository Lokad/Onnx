"""Run checker tests using pinned local dependencies, with no model execution."""
import prepare
import unittest
from pathlib import Path

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
