"""Load the already reviewed pointwise application prerequisite bindings locally."""
from source_scope import APP, load
validate = load('pointwise_application_prerequisites', APP/'prerequisites.py').validate
