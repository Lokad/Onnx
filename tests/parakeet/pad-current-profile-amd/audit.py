"""Run the unchanged full capture audit against the newly qualified product."""
import runpy
from run import PARENT

if __name__ == '__main__':
    runpy.run_path(str(PARENT / 'audit.py'), run_name='__main__')
