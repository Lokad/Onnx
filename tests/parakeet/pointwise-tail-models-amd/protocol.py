"""Use the original complete-model job and resource contract unchanged."""
from pathlib import Path
source = Path(__file__).resolve().parent.parent/'decoder-lstm-layout-models-amd/protocol.py'
exec(compile(source.read_text(), str(source), 'exec'))
