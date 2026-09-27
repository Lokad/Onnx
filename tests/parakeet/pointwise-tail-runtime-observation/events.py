"""Use the existing event reconciliation with this experiment's fixed counts."""
from adapters import adapted
exec(compile(adapted('events.py'), __file__, 'exec'))
