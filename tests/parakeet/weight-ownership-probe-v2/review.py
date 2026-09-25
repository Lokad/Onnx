"""Use every original review check on the corrected consumer and new captures."""
import importlib.util
import sys
from run import OLD_TOOLS,initial

loader=importlib.util.spec_from_file_location('ownership_original_review',OLD_TOOLS/'review.py')
review=importlib.util.module_from_spec(loader);loader.loader.exec_module(review)

if __name__=='__main__':
    initial()
    {'build':review.build,'capture':review.capture}[sys.argv[1]]()
