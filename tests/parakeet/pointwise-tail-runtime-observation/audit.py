"""Use the original complete diagnostic audit with pointwise-specific checks."""
from adapters import adapted
exec(compile(adapted('audit.py'), __file__, 'exec'))
