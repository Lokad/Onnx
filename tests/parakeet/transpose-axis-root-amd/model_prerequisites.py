"""Load the already reviewed transpose application prerequisite bindings locally."""
from source_scope import APP, load
validate = load('transpose_application_prerequisites', APP/'prerequisites.py').validate
