"""Load the already reviewed rational application prerequisite bindings locally."""
from source_scope import APP, load
validate = load('rational_application_prerequisites', APP/'prerequisites.py').validate
