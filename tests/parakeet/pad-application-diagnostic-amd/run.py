"""Reuse terminal-identity checked transport in a new, single-use namespace."""
from pathlib import Path
import sys
from scope import replace_once

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
path = ROOT/'tests/parakeet/pad-runtime-diagnostic-amd/run.py'
source = path.read_text(encoding='utf8')
source = source.replace('parakeet-pad-runtime-diagnostic-amd-20260923', 'parakeet-pad-application-diagnostic-amd-20260926')
source = source.replace('lokad-parakeet-pad-runtime-diagnostic-20260923', 'lokad-parakeet-pad-application-diagnostic-20260926')
source = replace_once(source, '>=12*1024**3 and psutil.disk_usage', '>=11*1024**3 and psutil.disk_usage')
source = replace_once(source, ").free>=3*1024**3", ").free>=2*1024**3")
# Include the immutable manifest/assets and the adapted consumer source in the
# retained collection. This changes transport coverage, not execution checks.
source = replace_once(source, "'reference','evidence','tools']", "'reference','evidence','tools','assets','source/consumer']")
exec(compile(source, str(path), 'exec'), globals())
