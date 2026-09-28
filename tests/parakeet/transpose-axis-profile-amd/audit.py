"""Reuse every public-result, phase, node, ownership and resource check."""
import runpy
from run import PARENT

if __name__ == '__main__':
    runpy.run_path(str(PARENT/'audit.py'), run_name='__main__')
