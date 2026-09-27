"""Execute the complete original root/package auditor unchanged."""
import runpy
from prepare import PARENT

if __name__ == '__main__': runpy.run_path(str(PARENT/'audit.py'), run_name='__main__')
