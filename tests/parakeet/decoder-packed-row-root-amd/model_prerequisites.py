"""Load the already reviewed prepared-row application prerequisite bindings locally."""
from source_scope import APP, load
validate = load('prepared_row_application_prerequisites', APP/'prerequisites.py').validate
