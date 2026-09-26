"""Apply the unchanged original screen auditor to the new isolated campaign."""
import importlib.util
import sys
import run

spec=importlib.util.spec_from_file_location('rational_screen_original_audit',run.OLD/'audit.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
assert original.BASE==run.BASE and original.REMOTE==run.REMOTE

if __name__=='__main__':{'build':original.build,'capture':original.capture}[sys.argv[1]]()
