"""Run the original complete graph auditor with this candidate's provenance."""
import runpy
from prepare import PARENT

if __name__ == '__main__':
    runpy.run_path(str(PARENT/'audit.py'), run_name='__main__')
