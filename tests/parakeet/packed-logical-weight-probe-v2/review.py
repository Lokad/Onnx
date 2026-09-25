"""Apply the unchanged review to the separately built Linux-guard correction."""
import importlib.util
import sys
import run

loader=importlib.util.spec_from_file_location('representation_original_review',run.OLD_TOOLS/'review.py')
review=importlib.util.module_from_spec(loader);loader.loader.exec_module(review)

if __name__=='__main__':{'build':review.build,'capture':review.capture}[sys.argv[1]]()
