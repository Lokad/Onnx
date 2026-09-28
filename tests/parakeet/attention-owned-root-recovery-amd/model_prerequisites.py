"""Load the already reviewed attention application prerequisite bindings locally."""
from source_scope import APP, load
validate = load('attention_application_prerequisites', APP/'prerequisites.py').validate
