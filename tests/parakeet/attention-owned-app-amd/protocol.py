"""Reuse all original application jobs and resource limits."""
from pathlib import Path
source = Path(__file__).resolve().parent.parent/'decoder-lstm-layout-app-amd/protocol.py'
exec(compile(source.read_text(), str(source), 'exec'))
